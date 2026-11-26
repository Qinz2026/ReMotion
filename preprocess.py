import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import DDIMScheduler, VideoToVideoSDPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth_img2img import preprocess_video
from omegaconf import OmegaConf
from torchvision.io import read_video
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import logging
from utilities.utils import save_video
import yaml


logging.set_verbosity_error()

def extract_attention_weighted_motion(features, attn_maps, masks):
    if features.dim() != 5:
        raise ValueError("features 必须是 5D Tensor, 形状应为 [B, C, F, H, W]")

    device = features.device
    dtype = features.dtype
    b, c, f, h, w = features.shape




    attn_maps = attn_maps.to(device=device)
    if attn_maps.dim() == 5:

        if attn_maps.shape[1] != 1:
            attn_maps = attn_maps.mean(dim=1, keepdim=True)

        if attn_maps.shape[2] != f:
            if attn_maps.shape[2] == 1:
                attn_maps = attn_maps.expand(-1, -1, f, -1, -1)
            else:
                raise ValueError("attn_maps 的帧维与 features 不一致")
    elif attn_maps.dim() == 4:

        if attn_maps.shape[1] == f and attn_maps.shape[2] == h and attn_maps.shape[3] == w:
            attn_maps = attn_maps.unsqueeze(1)
        elif attn_maps.shape[1] == 1 and attn_maps.shape[2] == h and attn_maps.shape[3] == w:

            attn_maps = attn_maps.unsqueeze(2).expand(-1, -1, f, -1, -1)
        elif attn_maps.shape[1] == h and attn_maps.shape[2] == w:

            attn_maps = attn_maps.unsqueeze(1).unsqueeze(2).expand(-1, -1, f, -1, -1)
        else:
            raise ValueError("attn_maps 形状无法与 features 对齐")
    elif attn_maps.dim() == 3:

        attn_maps = attn_maps.unsqueeze(1).unsqueeze(2).expand(-1, -1, f, -1, -1)
    else:
        raise ValueError("attn_maps 必须是 3D/4D/5D Tensor")




    masks = masks.to(device=device)
    if masks.dim() == 5:
        if masks.shape[1] != 1:

            mask_bool = (masks > 0.5).any(dim=1, keepdim=True)
        else:
            mask_bool = masks > 0.5
        if mask_bool.shape[2] != f:
            if mask_bool.shape[2] == 1:
                mask_bool = mask_bool.expand(-1, -1, f, -1, -1)
            else:
                raise ValueError("masks 的帧维与 features 不一致")
    elif masks.dim() == 4:
        if masks.shape[1] == f and masks.shape[2] == h and masks.shape[3] == w:
            mask_bool = (masks > 0.5).unsqueeze(1)
        elif masks.shape[1] == 1 and masks.shape[2] == h and masks.shape[3] == w:

            mask_bool = (masks > 0.5).unsqueeze(2).expand(-1, -1, f, -1, -1)
        elif masks.shape[1] == h and masks.shape[2] == w:
            mask_bool = (masks > 0.5).unsqueeze(1).unsqueeze(2).expand(-1, -1, f, -1, -1)
        else:
            raise ValueError("masks 形状无法与 features 对齐")
    elif masks.dim() == 3:
        mask_bool = (masks > 0.5).unsqueeze(1).unsqueeze(2).expand(-1, -1, f, -1, -1)
    else:
        raise ValueError("masks 必须是 3D/4D/5D Tensor")






    attn_masked = attn_maps.float()
    attn_masked = attn_masked.masked_fill(~mask_bool, float("-inf"))





    attn_flat = attn_masked.reshape(b, 1, f, -1)
    attn_weights = torch.softmax(attn_flat, dim=-1)

    attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
    attn_weights = attn_weights.to(dtype=dtype)




    features_flat = features.reshape(b, c, f, -1)
    motion_feat = (features_flat * attn_weights).sum(dim=-1)




    del attn_maps, masks, mask_bool, attn_masked, attn_flat, attn_weights, features_flat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return motion_feat

class Preprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = "cuda"
        self.config = config
        self.use_depth = False

        print("Loading video model")
        self.model_key = "cerspense/zeroscope_v2_576w"
        self.resolution = (576, 320)

        self.pipeline = VideoToVideoSDPipeline.from_pretrained(self.model_key, torch_dtype=torch.float16)
        self.pipeline = self.pipeline.to("cuda")
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_model_cpu_offload()

        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.scheduler = self.pipeline.scheduler
        print("video model loaded")

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(
            negative_prompt, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([text_embeddings, uncond_embeddings])
        return text_embeddings

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True):
        timesteps = list(reversed(self.scheduler.timesteps))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f"noisy_latents_{t}.pt"))
        torch.save(latent, os.path.join(save_path, f"noisy_latents_{t}.pt"))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(x.shape[0], 1, 1)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x - sigma * eps) / mu
                x = mu_prev * pred_x0 + sigma_prev * eps

        return x

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

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, inversion_prompt="", negative_prompt=""):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, negative_prompt)[[0]]
        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem))
            video = [Image.open(img).resize(self.resolution) for img in images]

        video = video[: self.config["max_number_of_frames"]]
        save_video([np.array(img) for img in video], str(Path(save_path) / f"original.mp4"))

        video = preprocess_video(video)
        bsz, channel, frames, width, height = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channel, width, height)
        latents = self.vae.config.scaling_factor * self.vae.encode(video.cuda().half()).latent_dist.sample()
        latents = latents[None, :].reshape((1, frames, latents.shape[1]) + latents.shape[2:]).permute(0, 2, 1, 3, 4)

        inverted_x = self.ddim_inversion(cond, latents, save_path, save_latents=True)

        if self.config["save_ddim_reconstruction"]:
            latent_reconstruction = self.ddim_sample(inverted_x, cond)
            decoded_latent = self.decode_latents(latent_reconstruction.half())
            video = tensor2vid(decoded_latent)
            save_video(video, str(Path(save_path) / f"result.mp4"))


def run(config):

    prompt_yaml_path = config.get("prompt_yaml_path", "prompt.yaml")
    if not os.path.exists(prompt_yaml_path):
        raise FileNotFoundError(f"找不到 prompt 配置文件: {prompt_yaml_path}")

    with open(prompt_yaml_path, 'r', encoding='utf-8') as f:
        prompts_data = yaml.safe_load(f)


    print("Initializing model...")
    model = Preprocess(config)

    base_video_path = Path(config.base_video_path)
    base_save_dir = Path(config.base_save_dir)
    prompt_prefix = config.get("prompt_prefix", "")


    for item in prompts_data:
        video_name = item['init_video'][0]
        source_prompt = item['source_prompt'][0]

        current_video_path = base_video_path / video_name
        current_save_path = base_save_dir / video_name
        full_prompt = prompt_prefix + source_prompt

        print(f"\n{'=' * 50}")
        print(f"准备处理视频: {video_name}")
        print(f"数据路径: {current_video_path}")
        print(f"Prompt: {full_prompt}")
        print(f"{'=' * 50}")



        if current_save_path.exists() and list(current_save_path.glob("noisy_latents_*.pt")):
            print(f"⏩ 目标文件夹已有处理结果，[{video_name}] 自动跳过！")
            continue



        if not current_video_path.exists():
            print(f"⚠️ 警告: 找不到源视频目录或文件 {current_video_path}，跳过。")
            continue

        os.makedirs(current_save_path, exist_ok=True)

        try:
            model.extract_latents(
                data_path=str(current_video_path),
                num_steps=config.n_timesteps,
                save_path=str(current_save_path),
                inversion_prompt=full_prompt,
                negative_prompt=config.negative_prompt,
            )
            print(f"✅ [{video_name}] 处理完成，保存在 {current_save_path}")
        except Exception as e:
            print(f"❌ 处理 [{video_name}] 时发生错误: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/preprocess.yaml")
    opt = parser.parse_args()


    config = OmegaConf.load(opt.config_path)

    run(config)
