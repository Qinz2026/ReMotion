"""ReMotion pipeline entry point; core model and guidance defaults live in pipeline_core.py."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
from pipeline_core import Guidance as CoreGuidance
from utilities.sparse_temporal import TemporalAttentionCapture, sparse_attention_loss


class Guidance(CoreGuidance):
    def __init__(self, config):
        super().__init__(config)
        self.sparse_weight = float(config.get("sparse_attention_weight", 0.0))
        self.sparse_capture = None
        self.sparse_diagnostics = []
        if self.sparse_weight:
            attention = self.unet.up_blocks[1].temp_attentions[1].transformer_blocks[0].attn1
            self.sparse_capture = TemporalAttentionCapture(attention, int(config["max_frames"]))
        self.latest_base_loss = self.latest_sparse_loss = None

    def _guidance_step(self, x, i, t):
        capture = self.sparse_capture
        if capture is not None:
            capture.clear()
            capture.active = True
        try:
            return super()._guidance_step(x, i, t)
        finally:
            if capture is not None:
                capture.active = False
                capture.clear()
            self.latest_base_loss = self.latest_sparse_loss = None

    def compute_guidance_loss(self, optimized_x, t, i, orig_features_pos):
        base = super().compute_guidance_loss(optimized_x, t, i, orig_features_pos)
        self.latest_base_loss = base
        if self.sparse_capture is None:
            return base
        capture = self.sparse_capture
        if capture.source is None or capture.target is None:
            raise RuntimeError("Source or target temporal attention missing")
        sparse, stats = sparse_attention_loss(capture.source, capture.target,
                                               self.video_masks, self.config)
        self.latest_sparse_loss = sparse
        self.sparse_diagnostics.append({"index": int(i), "timestep": int(t),
                                        "base_loss": float(base.detach()),
                                        "sparse_loss": float(sparse.detach()),
                                        "weight": self.sparse_weight, **stats})
        return base + self.sparse_weight * sparse

    def run(self):
        try:
            return super().run()
        finally:
            Path(self.config["output_path"], "sparse_diagnostics.json").write_text(
                json.dumps(self.sparse_diagnostics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, Path(config["output_path"]) / "config.yaml")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    guidance = Guidance(config) if float(config.get("sparse_attention_weight", 0.0)) else CoreGuidance(config)
    guidance.run()

