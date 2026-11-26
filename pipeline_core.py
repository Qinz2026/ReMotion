import argparse
import json
import copy
import os
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import TextToVideoSDPipeline, DDIMScheduler
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import logging

from utilities.guidance import (
    get_timesteps,
    register_time,
    register_guidance,
    calculate_losses,
    register_batch,
)
from utilities.method_improvements import project_motion_update
from utilities.amp_optimization import scaled_optimizer_step
from utilities.initialize_latent import (
    initialize_noisy_latent,
    load_source_latents_t,
)
from utilities.utils import (
    isinstance_str,
    clean_memory, save_video, seed_everything,
)


logging.set_verbosity_error()


class MyIdentityUnsqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x.unsqueeze(0)


class MyIdentity(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.train_mode = True

    def forward(self, x, *args, **kwargs):
        if self.train_mode:
            return x
        else:
            return self.module(x, *args, **kwargs)


class Guidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        model_key = config.get("model_path", "cerspense/zeroscope_v2_576w")
        self.resolution = (576, 320)

        print("Loading video model")

        self.video_pipe = TextToVideoSDPipeline.from_pretrained(
            model_key,
            torch_dtype=torch.float16,
        )
        self.video_pipe = self.video_pipe.to("cuda")
        self.video_pipe.scheduler = DDIMScheduler.from_config(self.video_pipe.scheduler.config)
        self.video_pipe.scheduler.set_timesteps(config["n_timesteps"], device="cuda")

        self.video_pipe.scheduler.timesteps, self.guidance_schedule = get_timesteps(
            self.video_pipe.scheduler,
            config["n_timesteps"],
            config["max_guidance_timestep"],
            config["min_guidance_timestep"],
        )
        delay_fraction = float(config.get("guidance_start_fraction", 0.0))
        if not 0.0 <= delay_fraction < 1.0:
            raise ValueError("guidance_start_fraction must be in [0,1)")
        delay = int(config["n_timesteps"] * delay_fraction)
        if delay:
            if delay >= len(self.guidance_schedule):
                raise ValueError("Guidance delay removes every guided timestep")
            self.guidance_schedule = self.guidance_schedule[delay:]
        self.video_pipe.enable_vae_slicing()
        self.vae = self.video_pipe.vae
        self.tokenizer = self.video_pipe.tokenizer
        self.text_encoder = self.video_pipe.text_encoder
        self.unet = self.video_pipe.unet
        self.scheduler = copy.deepcopy(self.video_pipe.scheduler)
        for component in (self.vae, self.text_encoder, self.unet):
            component.requires_grad_(False)
            component.eval()
        print("video model loaded")


        (
            self.video,
            self.video_masks,
            self.noisy_latent,
        ) = self.get_data()


        self.optim_lr = config["optim_lr"]
        self.optimization_steps = int(config["optimization_steps"])
        self.guidance_scaler = GradScaler()
        self.guidance_diagnostics = []
        self.scale_range = np.linspace(config["scale_range"][0], config["scale_range"][1], len(self.guidance_schedule))

        with torch.no_grad():
            self.text_embeds = self.video_pipe._encode_prompt(
                config["source_prompt"],
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            ).flip(
                0
            )

            self.guidance_embeds = self.video_pipe._encode_prompt(
                config["target_prompt"],
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=config["negative_prompt"],
            ).flip(
                0
            )

        self.current_latents = {
            t.item(): load_source_latents_t(t, self.config["latents_path"])[:, :, : config["max_frames"]].to(self.device)
            for t in self.scheduler.timesteps
        }
        for step, latent in self.current_latents.items():
            if latent.shape[2] != len(self.video):
                raise ValueError(
                    f"Source latent at timestep {step} has {latent.shape[2]} frames, "
                    f"but the video has {len(self.video)}."
                )


    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        batch_size, channels, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)

        image = self.vae.decode(latents).sample
        video = (
            image[None, :]
            .reshape(
                (
                    batch_size,
                    num_frames,
                    -1,
                )
                + image.shape[2:]
            )
            .permute(0, 2, 1, 3, 4)
        )

        video = video.float()
        return video

    def get_data(self):

        data_path = self.config["data_path"]
        images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
        images = sorted(images, key=lambda x: int(x.stem))[: self.config["max_frames"]]
        if not images:
            raise FileNotFoundError(f"No PNG or JPG frames found in {data_path}.")

        video = torch.stack(
            [ToTensor()(Image.open(img).convert("RGB").resize(self.resolution)) for img in images]
        ).cuda()

        masks_path = self.config["mask_path"]
        video_masks = []
        for mask_path in masks_path:
            mask_files = list(Path(mask_path).glob("*.png")) + list(Path(mask_path).glob("*.jpg"))
            mask_files = sorted(mask_files, key=lambda x: int(x.stem))[: len(images)]
            if len(mask_files) != len(images):
                raise ValueError(
                    f"Mask folder {mask_path} has {len(mask_files)} frames, but the source video has {len(images)}."
                )
            if [int(frame.stem) for frame in mask_files] != [int(frame.stem) for frame in images]:
                raise ValueError(f"Mask folder {mask_path} contains frame IDs that do not match the source video.")

            video_mask = torch.stack(
                [ToTensor()(Image.open(img).convert("L")) for img in mask_files]
            ).cuda()
            video_masks.append(video_mask > 0.5)
        if not video_masks:
            raise ValueError("mask_path must contain at least one background or subject mask folder.")
        video_masks = torch.stack(video_masks)


        if self.config["seed"] is None:
            seed = torch.randint(0, 1000000, (1,)).item()
            self.config["seed"] = seed
            with open(Path(self.config["output_path"], "seed.txt"), "w") as f:
                f.write(str(seed))

        seed_everything(self.config["seed"])
        video_for_latents = [np.array(Image.open(img).convert("RGB").resize(self.resolution)) for img in images]
        noisy_latent = initialize_noisy_latent(self, video_for_latents)

        return (
            video,
            video_masks,
            noisy_latent,
        )

    @staticmethod
    def my_pass(*args, **kwargs):
        if len(args) == 0:
            return kwargs["hidden_states"]
        else:
            return args[0]

    def change_mode(self, train=True):
        if getattr(self, "_guidance_truncated", False) == train:
            return
        if len(self.config["up_res_dict"]) != 0:
            index = max(self.config["up_res_dict"].keys())
            for i, block in enumerate(self.unet.up_blocks):
                if i > index:
                    if train:
                        self.unet.up_blocks[i].original_forward = self.unet.up_blocks[i].forward
                        self.unet.up_blocks[i].forward = self.my_pass
                    else:
                        self.unet.up_blocks[i].forward = self.unet.up_blocks[i].original_forward

        if self.unet.conv_norm_out:
            if train:
                self.unet.conv_norm_out.original_forward = self.unet.conv_norm_out.forward
                self.unet.conv_norm_out.forward = self.my_pass
            else:
                self.unet.conv_norm_out.forward = self.unet.conv_norm_out.original_forward

            if train:
                self.unet.conv_act.original_forward = self.unet.conv_act.forward
                self.unet.conv_act.forward = self.my_pass
            else:
                self.unet.conv_act.forward = self.unet.conv_act.original_forward

        if train:
            self.unet.conv_out.original_forward = self.unet.conv_out.forward
            self.unet.conv_out.forward = self.my_pass
        else:
            self.unet.conv_out.forward = self.unet.conv_out.original_forward

        self._guidance_truncated = train

    @torch.no_grad()
    def target_semantic_direction(self, x, t):
        register_batch(self, 2)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = self.unet(torch.cat([x, x]), t,
                                   encoder_hidden_states=self.guidance_embeds, return_dict=False)[0]
        conditional, unconditional = prediction.chunk(2)
        return (unconditional - conditional).float().detach()

    def guidance_step(self, x, i, t):
        try:
            return self._guidance_step(x, i, t)
        finally:
            self.change_mode(train=False)
            for _, module in self.unet.named_modules():
                if isinstance_str(module, ["ModuleWithConvGuidance", "ModuleWithGuidance"]):
                    module.saved_features = None

    def _guidance_step(self, x, i, t):
        semantic_direction = None
        projection_strength = float(self.config.get("semantic_projection_strength", 0.0))
        if projection_strength:
            semantic_direction = self.target_semantic_direction(x, t)
        register_batch(self, 1)
        module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
        policy = self.config.get("guidance_amp_policy", "legacy")
        if policy not in ("adaptive", "persistent", "legacy"):
            raise ValueError("guidance_amp_policy must be adaptive, persistent or legacy")
        scaler = GradScaler() if policy == "legacy" else self.guidance_scaler
        skipped = 0
        completed = 0
        max_skips = int(self.config.get("guidance_amp_max_skips", 24))
        if max_skips < 0:
            raise ValueError("guidance_amp_max_skips must be nonnegative")
        self.change_mode(train=True)
        optimized_x = x.clone().detach().requires_grad_(True)

        if self.config["with_lr_decay"]:
            guidance_index = int((self.guidance_schedule == t).nonzero(as_tuple=False)[0].item())
            lr = self.scale_range[guidance_index]
        else:
            lr = self.optim_lr

        optimizer = torch.optim.Adam([optimized_x], lr=lr)

        latents = self.current_latents[t.item()]

        with torch.no_grad():

            orig_features_pos = {}
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self.unet(latents, t, encoder_hidden_states=self.text_embeds[[0]], return_dict=False)[0]
            for _, module in self.unet.named_modules():
                if isinstance_str(module, module_names):
                    orig_features_pos[module.block_name] = module.saved_features[0]

        for _ in tqdm(range(self.optimization_steps)):
            # Retry the same requested update after a scale backoff. Adam state
            # and the latent stay unchanged on overflow.
            while True:
                optimizer.zero_grad(set_to_none=True)
                total_loss = self.compute_guidance_loss(optimized_x, t, i, orig_features_pos)
                if not bool(torch.isfinite(total_loss)):
                    raise FloatingPointError(f"Non-finite guidance loss at timestep {t.item()}")
                executed = scaled_optimizer_step(total_loss, optimizer, scaler)
                del total_loss
                for _, module in self.unet.named_modules():
                    if isinstance_str(module, module_names):
                        module.saved_features = None
                if executed:
                    completed += 1
                    break
                skipped += 1
                if policy != "adaptive":
                    break
                if skipped > max_skips:
                    raise FloatingPointError(f"AMP overflow did not recover at timestep {t.item()}; skips={skipped}")
        self.guidance_diagnostics.append({"timestep": int(t), "requested_updates": self.optimization_steps,
                                          "executed_updates": completed, "overflow_skips": skipped,
                                          "final_scale": scaler.get_scale(), "policy": policy})

        if semantic_direction is not None:
            update = project_motion_update(optimized_x.detach() - x, semantic_direction,
                                           strength=projection_strength)
            return (x + update).detach()
        return optimized_x.detach()

    def compute_guidance_loss(self, optimized_x, t, i, orig_features_pos):
        module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
        target_features_pos = {}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            self.unet(optimized_x, t, encoder_hidden_states=self.guidance_embeds[[0]].detach(), return_dict=False)[
                0]
        for _, module in self.unet.named_modules():
            if isinstance_str(module, module_names):
                target_features_pos[module.block_name] = module.saved_features[0]

        if not target_features_pos or orig_features_pos.keys() != target_features_pos.keys():
            raise RuntimeError("Guidance produced mismatched source/target feature layers")
        total_loss = 0

        for (orig_name, orig_features), (target_name, target_features) in zip(
                orig_features_pos.items(), target_features_pos.items()
        ):
            assert orig_name == target_name

            losses = calculate_losses(
                orig_features.detach(),
                target_features,
                self.video_masks,
                self.config,
                current_t=i,
                total_timesteps=len(self.scheduler.timesteps),
            )
            layer_weight = 1.0
            if "_temporal_attn" in orig_name:
                layer_weight = float(self.config.get("temporal_attention_loss_weight", 1.0))
            elif "_spatial_attn" in orig_name:
                layer_weight = float(self.config.get("spatial_attention_loss_weight", 1.0))
            total_loss += layer_weight * losses["total_loss"]

        if not target_features_pos or len(orig_features_pos) != len(target_features_pos):
            raise RuntimeError("Guidance produced no matched source/target feature layers.")

        return total_loss

    @torch.no_grad()
    def denoise_step(self, x, t):
        if t in self.guidance_schedule:
            self.change_mode(train=False)

        register_batch(self, 2)
        latent_model_input = torch.cat([x, x], dim=0)
        text_embed_input = self.guidance_embeds

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input, return_dict=False)[0]

        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

        bsz, channel, frames, width, height = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)

        denoised_latent = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        denoised_latent = denoised_latent[None, :].reshape(bsz, frames, channel, width, height).permute(0, 2, 1, 3, 4)

        return denoised_latent

    @torch.no_grad()
    def undo_step(self, sample, timestep):
        n = self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps

        for i in range(n):
            beta = self.scheduler.betas[timestep + i]
            noise = torch.randn_like(sample)
            sample = (1 - beta) ** 0.5 * sample + beta ** 0.5 * noise

        return sample

    def run(self):
        clean_memory()
        x = self.noisy_latent
        register_guidance(self)
        feature_layers = [{"name": module.block_name, "type": type(module).__name__}
                          for _, module in self.unet.named_modules()
                          if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"])]
        Path(self.config["output_path"], "feature_layers.json").write_text(
            json.dumps(feature_layers, indent=2) + "\n")

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            register_time(self, t.item())

            if t in self.guidance_schedule:
                x = self.guidance_step(x, i, t)
            x = self.denoise_step(x, t)
            if self.config["restart_sampling"] and t != self.scheduler.timesteps[-1]:
                x = self.undo_step(x, self.scheduler.timesteps[i + 1])
                x = self.denoise_step(x, t)

            clean_memory()
        Path(self.config["output_path"], "guidance_diagnostics.json").write_text(
            json.dumps(self.guidance_diagnostics, indent=2) + "\n")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            decoded_frames = self.decode_latents(x)
        if not bool(torch.isfinite(decoded_frames).all()):
            raise FloatingPointError("Decoded video contains non-finite values")
        decoded_frames = np.array(tensor2vid(decoded_frames))

        Path(self.config["output_path"], "result_frames").mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(decoded_frames):
            Image.fromarray(frame).save(Path(self.config["output_path"], "result_frames", f"{i:04d}.png"))

        save_video(decoded_frames, os.path.join(self.config["output_path"], "result.mp4"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/guidance.yaml")
    opt = parser.parse_args()


    config = OmegaConf.load(opt.config_path)

    print(f"\n{'=' * 50}\nStarting single process using: {opt.config_path}\n{'=' * 50}\n")


    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")


    try:
        guidance = Guidance(config)
        guidance.run()
    except Exception as e:
        print(f"Error occurred while processing: {e}")
        raise


    try:
        del guidance
    except NameError:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    clean_memory()
