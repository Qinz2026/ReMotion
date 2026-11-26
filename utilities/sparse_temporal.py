"""Research-only sparse temporal attention constraints, inspired by MotionClone.

This is an adaptation to frozen Zeroscope and the existing inversion pipeline,
not a reproduction of MotionClone and not an inversion-free implementation.
"""
import math
import torch
import torch.nn.functional as F
from utilities.guidance import _prepare_video_masks, _resolve_region_masks


def temporal_probabilities(query, key, heads):
    if query.ndim != 3 or query.shape != key.shape:
        raise ValueError("Expected matching [pixels, frames, heads*channels] projections")
    pixels, frames, channels = query.shape
    if channels % heads:
        raise ValueError("Projection channels must be divisible by heads")
    width = channels // heads
    q = query.reshape(pixels, frames, heads, width).transpose(1, 2).float()
    k = key.reshape(pixels, frames, heads, width).transpose(1, 2).float()
    return (q @ k.transpose(-1, -2) / math.sqrt(width)).softmax(dim=-1)


def sparse_attention_loss(source, target, masks, config):
    """Top-k on the source only; no output, target metric, or semantic label lookahead."""
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("Attention must have matching [pixels, heads, frames, frames] shapes")
    source = source.detach()
    pixels, heads, frames, keys = source.shape
    if frames != keys or frames < 2:
        raise ValueError("Temporal attention must be square with at least two frames")
    height = round(math.sqrt(pixels * 320 / 576))
    width = pixels // height
    if height * width != pixels:
        raise ValueError("Attention spatial resolution does not match 320:576 aspect ratio")
    resized = _prepare_video_masks(masks, frames, (height, width))
    subjects, background = _resolve_region_masks(resized, config)
    topk = int(config.get("sparse_topk", 1))
    if not 1 <= topk <= frames:
        raise ValueError("sparse_topk outside frame range")
    indices = source.topk(topk, dim=-1).indices
    errors = (target.gather(-1, indices) - source.gather(-1, indices)).square()
    mode = config.get("sparse_mode", "fixed")
    if mode not in ("fixed", "confidence"):
        raise ValueError("sparse_mode must be fixed or confidence")
    confidence = 1 - (-(source * source.clamp_min(1e-12).log()).sum(-1)) / math.log(frames)
    confidence = confidence.clamp_min(0.05).detach()
    foreground = subjects.any(0) if len(subjects) else torch.zeros_like(background)
    zero = target.sum() * 0.0

    def region_loss(region, other):
        # Query masks assign a source pixel/frame to exactly one subject or background.
        queries = region.reshape(frames, pixels).T[:, None, :, None]
        weights = queries.expand(-1, heads, -1, topk).float()
        if mode == "confidence":
            # Permit object entry/exit at a pixel, excluding keys occupied by other
            # source subjects. Entirely invisible subject frames cannot supply keys.
            visible = region.flatten(1).any(-1)
            allowed = ((~other) & visible[:, None, None]).reshape(frames, pixels).T
            allowed = allowed[:, None, None, :].expand(-1, heads, frames, -1)
            weights = weights * allowed.gather(-1, indices) * confidence[..., None]
        mass = weights.sum()
        value = (errors * weights).sum() / mass.clamp_min(1e-12)
        return value, bool(mass.detach() > 0)

    values = []
    for subject in subjects:
        value, valid = region_loss(subject, foreground & ~subject)
        if valid:
            values.append(value)
    subject_loss = torch.stack(values).mean() if values else zero
    background_loss, bg_valid = region_loss(background, foreground)
    if values and bg_valid:
        loss = 0.75 * subject_loss + 0.25 * background_loss
    elif values:
        loss = subject_loss
    else:
        loss = background_loss
    stats = {"source_confidence_mean": float(confidence.mean()),
             "subject_loss": float(subject_loss.detach()),
             "background_loss": float(background_loss.detach()),
             "valid_subjects": len(values), "attention_shape": list(source.shape)}
    return loss, stats


class TemporalAttentionCapture:
    """Observe real Q/K projections without replacing the original forward path."""
    def __init__(self, attention, frames):
        self.active = False
        self.frames = frames
        self.source = None
        self.target = None
        self.query = None
        self.key = None
        self.attention = attention
        self.handles = [
            attention.to_q.register_forward_hook(self._query),
            attention.to_k.register_forward_hook(self._key),
            attention.register_forward_hook(self._finish),
        ]

    def _query(self, module, args, output):
        if self.active:
            self.query = output

    def _key(self, module, args, output):
        if self.active:
            self.key = output

    def _finish(self, module, args, output):
        if not self.active:
            return
        if self.query is None or self.key is None:
            raise RuntimeError("Missing temporal Q/K capture")
        if self.query.shape[1] != self.frames:
            raise RuntimeError("Captured attention is not frame-to-frame self-attention")
        if module.norm_q is not None if hasattr(module, "norm_q") else False:
            raise RuntimeError("This capture adapter does not support Q normalization")
        if module.norm_k is not None if hasattr(module, "norm_k") else False:
            raise RuntimeError("This capture adapter does not support K normalization")
        probabilities = temporal_probabilities(self.query, self.key, module.heads)
        if torch.is_grad_enabled():
            self.target = probabilities
        else:
            self.source = probabilities.detach()
        self.query = self.key = None

    def clear(self):
        self.source = self.target = self.query = self.key = None

