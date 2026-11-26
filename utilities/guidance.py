from math import sqrt

import torch
import torch.nn.functional as F
from einops import rearrange

from utilities.utils import get_dynamic_soft_guidance, isinstance_str
from utilities.method_improvements import local_motion_consistency_loss


def masked_mean(tensor, mask):
    """Average spatial features inside a mask, returning zero for an empty mask."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError("mask must have shape [H,W], [B,H,W], or [B,1,H,W].")

    mask = mask.to(device=tensor.device, dtype=tensor.dtype)
    area = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    return (tensor * mask).sum(dim=(-2, -1), keepdim=True) / area


def resize_box(top_left: tuple, bottom_right: tuple, scale_x: float, scale_y: float) -> tuple:
    center_x = (top_left[0] + bottom_right[0]) / 2
    center_y = (top_left[1] + bottom_right[1]) / 2
    half_width = (bottom_right[0] - top_left[0]) / 2
    half_height = (bottom_right[1] - top_left[1]) / 2

    new_top_left = (center_x - half_width * scale_x, center_y - half_height * scale_y)
    new_bottom_right = (center_x + half_width * scale_x, center_y + half_height * scale_y)
    return new_top_left, new_bottom_right


def expand_mask_regions(mask_tensor, height_scale, width_scale):
    """Expand each [subject, frame] mask by scaling its bounding box."""
    if not isinstance(mask_tensor, torch.Tensor) or mask_tensor.dim() != 4:
        raise ValueError("mask_tensor must be a 4D tensor shaped as (subjects, frames, h, w).")

    expanded_mask = torch.zeros_like(mask_tensor, dtype=torch.bool)
    height, width = mask_tensor.shape[-2:]
    for subject_idx in range(mask_tensor.shape[0]):
        for frame_idx in range(mask_tensor.shape[1]):
            active_pixels = torch.nonzero(mask_tensor[subject_idx, frame_idx], as_tuple=False)
            if active_pixels.numel() == 0:
                continue

            y_min, y_max = active_pixels[:, 0].min(), active_pixels[:, 0].max()
            x_min, x_max = active_pixels[:, 1].min(), active_pixels[:, 1].max()
            (x1, y1), (x2, y2) = resize_box(
                (x_min, y_min), (x_max, y_max), width_scale, height_scale
            )
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width - 1, int(x2)), min(height - 1, int(y2))
            expanded_mask[subject_idx, frame_idx, y1 : y2 + 1, x1 : x2 + 1] = True

    return expanded_mask


new_mean = masked_mean
scale_bbox = resize_box
expand_true_regions = expand_mask_regions


def _as_video_features(features, name):
    if features.dim() == 5:
        if features.shape[0] != 1:
            raise ValueError(f"{name} currently supports a single video batch, got {features.shape[0]}.")
        features = features[0]
    if features.dim() != 4:
        raise ValueError(f"{name} must have shape [frames, channels, height, width].")
    return features


def _prepare_video_masks(masks, num_frames, spatial_size):
    """Convert [S,T,C,H,W] or [S,T,H,W] masks to feature-resolution booleans."""
    if masks.dim() == 5:
        if masks.shape[2] == 1:
            masks = masks[:, :, 0]
        else:
            masks = (masks > 0.5).any(dim=2)
    elif masks.dim() != 4:
        raise ValueError("masks must have shape [subjects, frames, height, width] or [subjects, frames, channels, height, width].")

    if masks.shape[1] != num_frames:
        raise ValueError(
            f"Mask frame count ({masks.shape[1]}) does not match feature frame count ({num_frames})."
        )

    num_subjects, _, mask_h, mask_w = masks.shape
    if num_subjects == 0:
        raise ValueError("At least one mask folder is required.")
    flat_masks = masks.float().reshape(num_subjects * num_frames, 1, mask_h, mask_w)
    resized = F.interpolate(flat_masks, size=spatial_size, mode="nearest")
    return resized.reshape(num_subjects, num_frames, *spatial_size) > 0.5


def _resolve_region_masks(masks, config):
    num_masks = masks.shape[0]
    background_index = config.get("background_mask_index", None)
    if background_index is not None:
        background_index = int(background_index)
        if background_index < 0:
            background_index += num_masks
        if not 0 <= background_index < num_masks:
            raise ValueError(
                f"background_mask_index={background_index} is out of range for {num_masks} mask folders."
            )

    configured_subjects = config.get("subject_mask_indices", None)
    if configured_subjects is None:
        subject_indices = [i for i in range(num_masks) if i != background_index]
    else:
        subject_indices = [int(i) for i in configured_subjects]
        invalid = [i for i in subject_indices if i < 0 or i >= num_masks]
        if invalid:
            raise ValueError(f"subject_mask_indices contains out-of-range indices: {invalid}.")
        if len(subject_indices) != len(set(subject_indices)):
            raise ValueError("subject_mask_indices contains duplicate indices.")
        if background_index in subject_indices:
            raise ValueError("The background mask cannot also be a subject mask.")

    subject_masks = masks[subject_indices]
    subject_masks = _make_subject_masks_exclusive(subject_masks)
    subject_union = subject_masks.any(dim=0) if subject_masks.shape[0] else torch.zeros_like(masks[0])

    if background_index is None:
        background_masks = ~subject_union
    else:
        background_masks = masks[background_index] & ~subject_union

    return subject_masks, background_masks


def _make_subject_masks_exclusive(subject_masks):
    if subject_masks.shape[0] <= 1:
        return subject_masks

    exclusive = []
    for subject_idx in range(subject_masks.shape[0]):
        other_indices = [i for i in range(subject_masks.shape[0]) if i != subject_idx]
        other_union = subject_masks[other_indices].any(dim=0)
        exclusive.append(subject_masks[subject_idx] & ~other_union)
    return torch.stack(exclusive, dim=0)


def mask_centroids(subject_masks):
    """Compute normalized geometric centers of the source mask tracks."""
    num_subjects, num_frames, height, width = subject_masks.shape
    device = subject_masks.device
    y_coords = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x_coords = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1)

    if num_subjects == 0:
        return (
            torch.zeros((0, num_frames, 2), device=device, dtype=torch.float32),
            torch.zeros((0, num_frames), device=device, dtype=torch.bool),
        )

    weights = subject_masks.float()
    area = weights.sum(dim=(-2, -1))
    centers = (weights[..., None] * coords).sum(dim=(-3, -2)) / area.clamp_min(1.0)[..., None]
    return centers, area > 0


def masked_feature_centroids(features, subject_masks, temperature=1.0):
    """Estimate differentiable 2D feature centers inside tracked subject masks.

    Feature-vector energy supplies soft spatial weights. Coordinates are normalized
    to [-1, 1], so the resulting track is independent of feature-map resolution.
    """
    if temperature <= 0:
        raise ValueError("feature centroid temperature must be greater than zero.")

    num_frames, channels, height, width = features.shape
    if subject_masks.shape[1:] != (num_frames, height, width):
        raise ValueError("subject_masks and features have incompatible frame or spatial dimensions.")

    num_subjects = subject_masks.shape[0]
    if num_subjects == 0:
        return (
            features.new_zeros((0, num_frames, 2), dtype=torch.float32),
            torch.zeros((0, num_frames), device=features.device, dtype=torch.bool),
        )

    energy = features.float().square().mean(dim=1).clamp_min(1e-12).sqrt()
    energy = torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
    energy = energy.reshape(1, num_frames, height * width)
    mask_weights = subject_masks.reshape(num_subjects, num_frames, height * width).float()
    areas = mask_weights.sum(dim=-1, keepdim=True)
    valid_frames = areas.squeeze(-1) > 0

    means = (energy * mask_weights).sum(dim=-1, keepdim=True) / areas.clamp_min(1.0)
    variance = ((energy - means).square() * mask_weights).sum(dim=-1, keepdim=True) / areas.clamp_min(1.0)
    # An epsilon outside sqrt still gives an infinite derivative when variance=0.
    normalized_energy = (energy - means) / (variance + 1e-6).sqrt()
    logits = (normalized_energy / float(temperature)).expand(num_subjects, -1, -1)
    logits = logits.masked_fill(mask_weights == 0, -1e4)
    weights = torch.softmax(logits, dim=-1) * mask_weights
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    y_coords = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=torch.float32)
    x_coords = torch.linspace(-1.0, 1.0, width, device=features.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1).reshape(height * width, 2)
    centers = (weights[..., None] * coords[None, None, :, :]).sum(dim=-2)
    return centers, valid_frames


def prototype_feature_centroids(source_features, target_features, subject_masks, temperature=0.05):
    """Locate source subject feature prototypes in the entire target feature map.

    Unlike a centroid restricted to the moving source mask, this estimate can
    detect a target feature at a different location. It remains a proxy for
    object location and must be checked against tracks on generated videos.
    """
    if temperature <= 0:
        raise ValueError("prototype_temperature must be greater than zero.")
    frames, channels, height, width = source_features.shape
    if target_features.shape != source_features.shape or subject_masks.shape[1:] != (frames, height, width):
        raise ValueError("Prototype tracking requires aligned source/target features and masks.")
    subjects = subject_masks.shape[0]
    if subjects == 0:
        return target_features.new_zeros((0, frames, 2)), subject_masks.new_zeros((0, frames))

    weights = subject_masks.to(dtype=source_features.dtype)
    areas = weights.sum(dim=(-2, -1))
    prototypes = torch.einsum("sthw,tchw->stc", weights, source_features)
    prototypes = prototypes / areas.clamp_min(1.0)[..., None]
    valid = (areas > 0) & (prototypes.norm(dim=-1) > 1e-6)
    prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
    normalized_target = F.normalize(target_features, dim=1, eps=1e-6)
    scores = torch.einsum("stc,tchw->sthw", prototypes, normalized_target)
    probabilities = (scores.flatten(-2) / float(temperature)).softmax(dim=-1)

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=target_features.device, dtype=target_features.dtype),
        torch.linspace(-1.0, 1.0, width, device=target_features.device, dtype=target_features.dtype),
        indexing="ij",
    )
    coords = torch.stack((xx, yy), dim=-1).reshape(height * width, 2)
    centers = probabilities @ coords
    return centers, valid


def trajectory_alignment_loss(source_centers, target_centers, valid_frames, velocity_weight=0.5):
    """Align each subject's path relative to its first visible frame and velocity."""
    zero = target_centers.sum() * 0.0
    subject_losses = []
    for subject_idx in range(source_centers.shape[0]):
        indices = torch.nonzero(valid_frames[subject_idx], as_tuple=False).flatten()
        if indices.numel() < 2:
            continue

        first_index = indices[0]
        source_path = source_centers[subject_idx, indices] - source_centers[subject_idx, first_index]
        target_path = target_centers[subject_idx, indices] - target_centers[subject_idx, first_index]
        position_loss = F.smooth_l1_loss(target_path, source_path)

        frame_gaps = (indices[1:] - indices[:-1]).float().unsqueeze(-1).clamp_min(1.0)
        source_velocity = (source_centers[subject_idx, indices[1:]] - source_centers[subject_idx, indices[:-1]]) / frame_gaps
        target_velocity = (target_centers[subject_idx, indices[1:]] - target_centers[subject_idx, indices[:-1]]) / frame_gaps
        velocity_loss = F.smooth_l1_loss(target_velocity, source_velocity)
        subject_losses.append(position_loss + float(velocity_weight) * velocity_loss)

    return torch.stack(subject_losses).mean() if subject_losses else zero


def interaction_relation_loss(
    source_centers,
    target_centers,
    valid_frames,
    velocity_weight=0.5,
    order_margin=0.02,
    order_weight=0.25,
):
    """Align signed pairwise relative tracks, including left/right/up/down ordering."""
    zero = target_centers.sum() * 0.0
    pair_losses = []
    num_subjects = source_centers.shape[0]
    for first in range(num_subjects):
        for second in range(first + 1, num_subjects):
            indices = torch.nonzero(valid_frames[first] & valid_frames[second], as_tuple=False).flatten()
            if indices.numel() < 2:
                continue

            source_relative = source_centers[first, indices] - source_centers[second, indices]
            target_relative = target_centers[first, indices] - target_centers[second, indices]
            relation_loss = F.smooth_l1_loss(target_relative, source_relative)

            frame_gaps = (indices[1:] - indices[:-1]).float().unsqueeze(-1).clamp_min(1.0)
            source_velocity = (source_relative[1:] - source_relative[:-1]) / frame_gaps
            target_velocity = (target_relative[1:] - target_relative[:-1]) / frame_gaps
            velocity_loss = F.smooth_l1_loss(target_velocity, source_velocity)

            ordered_axes = source_relative.abs() > float(order_margin)
            order_penalty = F.relu(float(order_margin) - source_relative.sign() * target_relative)
            order_loss = (order_penalty * ordered_axes).sum() / ordered_axes.sum().clamp_min(1)

            pair_losses.append(
                relation_loss
                + float(velocity_weight) * velocity_loss
                + float(order_weight) * order_loss
            )

    return torch.stack(pair_losses).mean() if pair_losses else zero


def _compute_feature_motion_losses(
    source_features,
    target_features,
    subject_masks,
    background_masks,
    config,
    current_t=None,
    total_timesteps=None,
):
    """Retain masked feature-difference guidance as a weaker appearance/motion term."""
    zero = target_features.sum() * 0.0
    if source_features.shape[0] < 2:
        return zero, zero, zero

    scale_h = float(config.get("h_rescale", 1.0))
    scale_w = float(config.get("w_rescale", 1.0))
    if scale_h == 1.0 and scale_w == 1.0:
        expanded_subject_masks = subject_masks
    else:
        expanded_subject_masks = expand_mask_regions(subject_masks.float(), scale_h, scale_w)
        expanded_subject_masks = _make_subject_masks_exclusive(expanded_subject_masks)

    subject_transition_losses = []
    background_transition_losses = []
    for frame_idx in range(1, source_features.shape[0]):
        frame_subject_losses = []
        for subject_idx in range(subject_masks.shape[0]):
            source_roi = subject_masks[subject_idx, frame_idx] | subject_masks[subject_idx, frame_idx - 1]
            target_roi = expanded_subject_masks[subject_idx, frame_idx] | expanded_subject_masks[subject_idx, frame_idx - 1]
            source_velocity = masked_mean(source_features[frame_idx : frame_idx + 1], source_roi)
            source_velocity = source_velocity - masked_mean(source_features[frame_idx - 1 : frame_idx], source_roi)
            target_velocity = masked_mean(target_features[frame_idx : frame_idx + 1], target_roi)
            target_velocity = target_velocity - masked_mean(target_features[frame_idx - 1 : frame_idx], target_roi)
            frame_subject_losses.append(
                compute_direction_aware_loss(
                    source_velocity,
                    target_velocity,
                    margin=float(config.get("direction_aware_margin", 0.1)),
                    lambda_dir=float(config.get("direction_aware_lambda_dir", 1.0)),
                    lambda_mag=float(config.get("direction_aware_lambda_mag", 0.0)),
                )
            )

        subject_loss = torch.stack(frame_subject_losses).mean() if frame_subject_losses else zero
        subject_transition_losses.append(subject_loss)

        background_roi = background_masks[frame_idx] | background_masks[frame_idx - 1]
        source_velocity = masked_mean(source_features[frame_idx : frame_idx + 1], background_roi)
        source_velocity = source_velocity - masked_mean(source_features[frame_idx - 1 : frame_idx], background_roi)
        target_velocity = masked_mean(target_features[frame_idx : frame_idx + 1], background_roi)
        target_velocity = target_velocity - masked_mean(target_features[frame_idx - 1 : frame_idx], background_roi)
        background_transition_losses.append(
            compute_direction_aware_loss(
                source_velocity,
                target_velocity,
                margin=float(config.get("direction_aware_margin", 0.1)),
                lambda_dir=float(config.get("direction_aware_lambda_dir", 1.0)),
                lambda_mag=float(config.get("direction_aware_lambda_mag", 0.0)),
            )
        )

    subject_loss = torch.stack(subject_transition_losses).mean() if subject_transition_losses else zero
    background_loss = torch.stack(background_transition_losses).mean() if background_transition_losses else zero
    if current_t is not None and total_timesteps is not None:
        combined_loss = get_dynamic_soft_guidance(
            delta_bg=background_loss,
            delta_subj=subject_loss,
            current_t=current_t,
            total_timesteps=total_timesteps,
            alpha_start=float(config.get("soft_guidance_alpha_start", 0.8)),
            alpha_end=float(config.get("soft_guidance_alpha_end", 0.1)),
            power=float(config.get("soft_guidance_power", 1.0)),
        )
    else:
        combined_loss = subject_loss + background_loss
    return subject_loss, background_loss, combined_loss


def calculate_losses(orig_features, target_features, masks, config, current_t=None, total_timesteps=None):
    """Calculate trajectory, pairwise interaction, and auxiliary feature losses.

    `masks` are ordered like config.mask_path. An explicit background mask is
    excluded from the subject set and is never treated as a moving subject.
    """
    source_features = _as_video_features(orig_features, "orig_features").detach().float()
    generated_features = _as_video_features(target_features, "target_features").float()
    if source_features.shape != generated_features.shape:
        raise ValueError(
            f"Source and target feature shapes must match, got {tuple(source_features.shape)} "
            f"and {tuple(generated_features.shape)}."
        )

    source_masks = _prepare_video_masks(masks, source_features.shape[0], source_features.shape[-2:])
    subject_masks, background_masks = _resolve_region_masks(source_masks, config)
    zero = generated_features.sum() * 0.0
    if source_features.shape[0] < 2:
        return {
            "subject_trajectory_loss": zero,
            "interaction_relation_loss": zero,
            "subject_feature_motion_loss": zero,
            "background_feature_motion_loss": zero,
            "semantic_velocity_loss": zero,
            "features_diff_loss": zero,
            "total_loss": zero,
        }

    # The source trajectory comes from tracked masks. The target estimate can
    # search the whole feature map, avoiding a tautology from fixed source ROIs.
    source_centers, source_valid = mask_centroids(subject_masks)
    estimator = config.get("trajectory_estimator", "prototype")
    if estimator == "prototype":
        target_centers, target_valid = prototype_feature_centroids(
            source_features, generated_features, subject_masks,
            temperature=float(config.get("prototype_temperature", 0.05)),
        )
        if config.get("prototype_source_calibration", False):
            # Correct frame-dependent localization bias using a detached source
            # reference. Identical source/target features then recover the mask
            # trajectory, rather than introducing a spurious motion penalty.
            source_proxy, _ = prototype_feature_centroids(
                source_features, source_features, subject_masks,
                temperature=float(config.get("prototype_temperature", 0.05)),
            )
            target_centers = target_centers + (source_centers - source_proxy).detach()
    elif estimator == "masked_energy":
        target_centers, target_valid = masked_feature_centroids(
            generated_features, subject_masks,
            temperature=float(config.get("feature_centroid_temperature", 1.0)),
        )
    else:
        raise ValueError(f"Unknown trajectory_estimator: {estimator}")
    valid_frames = source_valid & target_valid

    trajectory_loss = trajectory_alignment_loss(
        source_centers,
        target_centers,
        valid_frames,
        velocity_weight=float(config.get("trajectory_velocity_weight", 0.5)),
    )
    relation_loss = interaction_relation_loss(
        source_centers,
        target_centers,
        valid_frames,
        velocity_weight=float(config.get("interaction_velocity_weight", 0.5)),
        order_margin=float(config.get("interaction_order_margin", 0.02)),
        order_weight=float(config.get("interaction_order_weight", 0.25)),
    )

    subject_feature_loss, background_feature_loss, semantic_velocity_loss = _compute_feature_motion_losses(
        source_features,
        generated_features,
        subject_masks,
        background_masks,
        config,
        current_t=current_t,
        total_timesteps=total_timesteps,
    )

    trajectory_weight = float(config.get("trajectory_loss_weight", 1.0))
    interaction_weight = float(config.get("interaction_loss_weight", 0.5))
    feature_weight = float(config.get("semantic_velocity_loss_weight", 0.25))
    features_diff_loss = (
        trajectory_weight * trajectory_loss
        + interaction_weight * relation_loss
        + feature_weight * semantic_velocity_loss
    )
    local_motion_loss = zero
    local_weight = float(config.get("local_motion_loss_weight", 0.0))
    if local_weight:
        local_motion_loss = local_motion_consistency_loss(
            source_features, generated_features, subject_masks, background_masks,
            radius=int(config.get("local_motion_radius", 2)),
            temperature=float(config.get("local_motion_temperature", 0.1)),
        )
        features_diff_loss = features_diff_loss + local_weight * local_motion_loss
    total_loss = float(config.get("features_diff_loss_weight", 1.0)) * features_diff_loss

    return {
        "subject_trajectory_loss": trajectory_loss,
        "interaction_relation_loss": relation_loss,
        "subject_feature_motion_loss": subject_feature_loss,
        "background_feature_motion_loss": background_feature_loss,
        "semantic_velocity_loss": semantic_velocity_loss,
        "local_motion_loss": local_motion_loss,
        "features_diff_loss": features_diff_loss,
        "total_loss": total_loss,
    }


def compute_direction_aware_loss(delta_src, delta_tgt, margin=0.1, lambda_dir=1.0, lambda_mag=0.5):
    """Compare feature-change directions while avoiding a false penalty for two zero vectors."""
    src_items = list(delta_src) if isinstance(delta_src, (list, tuple)) else [delta_src]
    tgt_items = list(delta_tgt) if isinstance(delta_tgt, (list, tuple)) else [delta_tgt]

    if len(src_items) != len(tgt_items):
        if len(src_items) == 1:
            src_items = src_items * len(tgt_items)
        elif len(tgt_items) == 1:
            tgt_items = tgt_items * len(src_items)
        else:
            raise ValueError("delta_src and delta_tgt cannot be broadcast to the same subject count.")

    total_loss = None
    for source_delta, target_delta in zip(src_items, tgt_items):
        if not torch.is_tensor(source_delta) or not torch.is_tensor(target_delta):
            raise TypeError("delta_src and delta_tgt must be tensors or tensor lists.")

        source_flat = _flatten_feature_batch(source_delta.float())
        target_flat = _flatten_feature_batch(target_delta.float())
        if source_flat.shape[0] != target_flat.shape[0]:
            if source_flat.shape[0] == 1:
                source_flat = source_flat.expand(target_flat.shape[0], -1)
            elif target_flat.shape[0] == 1:
                target_flat = target_flat.expand(source_flat.shape[0], -1)
            else:
                raise ValueError("delta_src and delta_tgt batch dimensions cannot be broadcast.")

        cosine_sim = F.cosine_similarity(source_flat, target_flat, dim=1, eps=1e-8)
        moving = (source_flat.norm(p=2, dim=1) > 1e-8) | (target_flat.norm(p=2, dim=1) > 1e-8)
        direction_loss = (1.0 - cosine_sim)[moving].mean() if bool(moving.any()) else source_flat.sum() * 0.0
        # Compare speed (norm) separately from direction, with a tolerance.
        source_speed = torch.linalg.vector_norm(source_flat, dim=1)
        target_speed = torch.linalg.vector_norm(target_flat, dim=1)
        magnitude_loss = F.relu((source_speed - target_speed).abs() - margin).square().mean()
        pair_loss = lambda_dir * direction_loss + lambda_mag * magnitude_loss
        total_loss = pair_loss if total_loss is None else total_loss + pair_loss

    if total_loss is None:
        raise ValueError("delta_src and delta_tgt must contain at least one tensor pair.")
    return total_loss / len(src_items)


def _flatten_feature_batch(features):
    if features.dim() == 0:
        return features.view(1, 1)
    if features.dim() == 1:
        return features.unsqueeze(0)
    return features.reshape(features.shape[0], -1)


def get_timesteps(scheduler, num_inference_steps, max_guidance_timestep, min_guidance_timestep):
    init_timestep = min(int(num_inference_steps * max_guidance_timestep), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = int(num_inference_steps * min_guidance_timestep)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if t_end > 0:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order : -t_end * scheduler.order]
    else:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order :]
    return timesteps, guidance_schedule


def register_time(model, t):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "t", t)


def register_batch(model, b):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "b", b)


def register_guidance(model):
    guidance_schedule = model.guidance_schedule
    num_frames = model.video.shape[0]
    h = model.video.shape[2]
    w = model.video.shape[3]

    class ModuleWithConvGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "spatial_convolution",
            ]
            self.module_type = module_type
            if self.module_type == "spatial_convolution":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config
            self.saved_features = None

        def forward(self, input_tensor, temb):
            hidden_states = input_tensor

            hidden_states = self.module.norm1(hidden_states)
            hidden_states = self.module.nonlinearity(hidden_states)

            if self.module.upsample is not None:
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.module.upsample(input_tensor)
                hidden_states = self.module.upsample(hidden_states)
            elif self.module.downsample is not None:
                input_tensor = self.module.downsample(input_tensor)
                hidden_states = self.module.downsample(hidden_states)

            hidden_states = self.module.conv1(hidden_states)

            if temb is not None:
                temb = self.module.time_emb_proj(self.module.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.module.norm2(hidden_states)

            if temb is not None and self.module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.module.nonlinearity(hidden_states)
            hidden_states = self.module.dropout(hidden_states)
            hidden_states = self.module.conv2(hidden_states)

            if self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    hidden_states,
                    f"{self.starting_shape} -> b t d h w",
                    t=self.num_frames,
                )

            if self.module.conv_shortcut is not None:
                input_tensor = self.module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.module.output_scale_factor

            if not self.config["guidance_before_res"] and (self.t in self.guidance_schedule):
                self.saved_features = rearrange(
                    output_tensor,
                    f"{self.starting_shape} -> b t d h w",
                    t=self.num_frames,
                )

            return output_tensor

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
                "spatial_attention",
                "temporal_convolution",
                "upsampler",
            ]
            self.module_type = module_type
            if self.module_type == "temporal_attention":
                self.starting_shape = "(b h w) t d"
            elif self.module_type == "spatial_attention":
                self.starting_shape = "(b t) (h w) d"
            elif self.module_type == "temporal_convolution":
                self.starting_shape = "(b t) d h w"
            elif self.module_type == "upsampler":
                self.starting_shape = "(b t) d h w"
            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config
            self.saved_features = None

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            if self.module_type == "temporal_attention":
                size = out.shape[0] // self.b
            elif self.module_type == "spatial_attention":
                size = out.shape[1]
            elif self.module_type in ["temporal_convolution", "upsampler"]:
                size = out.shape[2] * out.shape[3]

            if self.t in self.guidance_schedule:
                h = int(sqrt(size * self.h / self.w))
                w = int(sqrt(size * self.h / self.w) * self.w / self.h)
                self.saved_features = rearrange(
                    out,
                    f"{self.starting_shape} -> b t d h w",
                    t=self.num_frames,
                    h=h,
                    w=w,
                )

            return out

    up_res_dict = model.config["up_res_dict"]
    for res in up_res_dict:
        module = model.unet.up_blocks[res]
        samplers = module.upsamplers
        if model.config["use_upsampler_features"]:
            if samplers is not None:
                for i in range(len(samplers)):
                    submodule = samplers[i]
                    samplers[i] = ModuleWithGuidance(
                        submodule,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=f"decoder_res{res}_upsampler",
                        config=model.config,
                        module_type="upsampler",
                    )
        for block in up_res_dict[res]:
            block_name = f"decoder_res{res}_block{block}"
            if model.config["use_conv_features"]:
                block_name_conv = f"{block_name}_spatial_convolution"
                submodule = module.resnets[block]
                module.resnets[block] = ModuleWithConvGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="spatial_convolution",
                )

            if model.config["use_temp_conv_features"]:
                block_name_conv = f"{block_name}_temporal_convolution"
                submodule = module.temp_convs[block]
                module.temp_convs[block] = ModuleWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=model.config,
                    module_type="temporal_convolution",
                )

            if res == 0 or model.config.get("enable_attention_guidance", False):
                if model.config["use_spatial_attention_features"] and hasattr(module, "attentions"):
                    block_name_spatial = f"{block_name}_spatial_attn1"
                    submodule = module.attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=block_name_spatial,
                        config=model.config,
                        module_type="spatial_attention",
                    )
                if model.config["use_temporal_attention_features"] and hasattr(module, "temp_attentions"):
                    submodule = module.temp_attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    block_name_temp = f"{block_name}_temporal_attn1"
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )
                    block_name_temp = f"{block_name}_temporal_attn2"
                    submodule.attn2 = ModuleWithGuidance(
                        submodule.attn2,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=model.config,
                        module_type="temporal_attention",
                    )
