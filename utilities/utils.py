import gc
import random
from typing import List, Union

import numpy as np
import torch
from torchvision.io import write_video

VIDEO_CODEC = "libx264rgb"
VIDEO_OPTIONS = {
    "crf": "0",
    "preset": "ultrafast",
    "threads": "1",
}


def save_video(video, path):
    write_video(
        path,
        video,
        fps=10,
        video_codec=VIDEO_CODEC,
        options=VIDEO_OPTIONS,
    )


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get_dynamic_soft_guidance(
    delta_bg,
    delta_subj,
    current_t,
    total_timesteps,
    alpha_start=0.8,
    alpha_end=0.1,
    power=1.0,
):

    if torch.is_tensor(current_t):
        if current_t.numel() == 1:
            current_t_val = float(current_t.item())
        else:
            current_t_val = float(current_t.float().mean().item())
    else:
        current_t_val = float(current_t)

    total_steps = float(max(int(total_timesteps) - 1, 1))
    progress = max(0.0, min(1.0, current_t_val / total_steps))

    if power is None or power <= 0:
        power = 1.0
    progress = progress ** float(power)

    blend_weight = float(alpha_start) + (float(alpha_end) - float(alpha_start)) * progress
    blend_weight = max(0.0, min(1.0, blend_weight))

    if torch.is_tensor(delta_bg):
        blend_weight_tensor = torch.tensor(
            blend_weight,
            device=delta_bg.device,
            dtype=delta_bg.dtype,
        )
        fused = blend_weight_tensor * delta_bg + (1.0 - blend_weight_tensor) * delta_subj
        del blend_weight_tensor
    else:
        fused = blend_weight * delta_bg + (1.0 - blend_weight) * delta_subj

    return fused


def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    if isinstance(cls_name, str):
        for parent_cls in x.__class__.__mro__:
            if parent_cls.__name__ == cls_name:
                return True
    else:
        for parent_cls in x.__class__.__mro__:
            if parent_cls.__name__ in cls_name:
                return True
    return False
