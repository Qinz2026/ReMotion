"""Optional motion-guidance components; defaults preserve the baseline path."""
import math
import torch
import torch.nn.functional as F


def correlate_temporal_noise(noise, correlation, seed):
    correlation=float(correlation)
    if not 0.0 <= correlation <= 1.0:
        raise ValueError('noise_temporal_correlation must be in [0,1]')
    if noise.ndim!=5:raise ValueError('Noise must be [B,C,T,H,W]')
    if correlation==0:return noise
    shape=(noise.shape[0],noise.shape[1],1,*noise.shape[-2:])
    generator=torch.Generator(device=noise.device).manual_seed(int(seed)+1000003)
    shared=torch.randn(shape,device=noise.device,dtype=noise.dtype,generator=generator)
    return math.sqrt(1-correlation)*noise+math.sqrt(correlation)*shared


def project_motion_update(delta, semantic_direction, strength=1.0):
    """Remove the component opposing the target CFG score direction, per frame."""
    strength=float(strength)
    if not 0.0<=strength<=1.0:raise ValueError('Projection strength must be in [0,1]')
    if delta.shape!=semantic_direction.shape or delta.ndim!=5:
        raise ValueError('Latent update and semantic direction must have matching [B,C,T,H,W] shapes')
    if strength==0:return delta
    direction=semantic_direction.detach().float()
    dimensions=(1,3,4)
    dot=(delta.float()*direction).sum(dim=dimensions,keepdim=True)
    norm=direction.square().sum(dim=dimensions,keepdim=True).clamp_min(1e-12)
    opposing=(dot/norm).clamp_max(0.0)
    return delta-strength*opposing*direction


def local_temporal_correlations(features, radius=2, temperature=0.1):
    """Local frame-to-frame correspondence probabilities, invariant to channel rotation."""
    if features.ndim!=4 or features.shape[0]<2:
        raise ValueError('Features must be [T,C,H,W] with at least two frames')
    if radius<1 or temperature<=0:raise ValueError('Invalid correlation radius/temperature')
    normalized=F.normalize(features.float(),dim=1,eps=1e-6)
    first=normalized[:-1]
    second=F.pad(normalized[1:],(radius,radius,radius,radius))
    height,width=features.shape[-2:]
    scores=[];valid=[]
    yy,xx=torch.meshgrid(torch.arange(height,device=features.device),torch.arange(width,device=features.device),indexing='ij')
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            shifted=second[:,:,radius+dy:radius+dy+height,radius+dx:radius+dx+width]
            scores.append((first*shifted).sum(dim=1))
            valid.append((yy+dy>=0)&(yy+dy<height)&(xx+dx>=0)&(xx+dx<width))
    logits=torch.stack(scores,dim=1)/float(temperature)
    valid=torch.stack(valid,dim=0)
    return logits.masked_fill(~valid[None],-1e4).log_softmax(dim=1)


def local_motion_consistency_loss(source,target,subject_masks,background_masks,radius=2,temperature=0.1):
    source_log=local_temporal_correlations(source.detach(),radius,temperature)
    target_log=local_temporal_correlations(target,radius,temperature)
    source_prob=source_log.exp()
    error=(source_prob*(source_log-target_log)).sum(dim=1)
    subjects=subject_masks.any(dim=0) if len(subject_masks) else torch.zeros_like(background_masks)
    subject_valid=subjects[:-1].float()
    background_valid=background_masks[:-1].float()
    def weighted(mask):
        return (error*mask).sum()/mask.sum().clamp_min(1)
    return 0.75*weighted(subject_valid)+0.25*weighted(background_valid)
