"""AMP update accounting for short latent-optimization loops."""
import torch


def scaled_optimizer_step(loss, optimizer, scaler):
    """Return whether AMP executed Adam; overflow must not consume a requested step.

    GradScaler.step returns Adam.step's None on both success and overflow, so use
    the documented scale backoff on overflow rather than that return value.
    """
    scaler.scale(loss).backward()
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale
