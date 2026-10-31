import torch
import numpy as np

import os


def get_spectrum(latent):
    fft = torch.fft.fft2(latent)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    return fft_shifted


def get_radius_map(H, W, device):
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    cy, cx = H//2, W//2
    r = torch.sqrt((x - cx).float()**2 + (y - cy).float()**2)
    return r


def make_gaussian_lowpass_filter(H, W, device, gaussian_sigma):
    r = get_radius_map(H, W, device)
    sigma = gaussian_sigma * r.max()
    gaussian_filter = torch.exp(-(r**2) / (2.0 * (sigma**2) + 1e-12))
    return gaussian_filter


def make_box_lowpass_filter(H, W, device, cutoff_ratio):
    r = get_radius_map(H, W, device)
    cutoff_radius = cutoff_ratio * r.max()
    box_filter = (r <= cutoff_radius).float()
    return box_filter


def fourier_swapping(
    source_latent,
    target_latent,
    step_index,
    total_timesteps,
    gaussian_lambda=0.3,
    phase_strength=1.0,
    decay_type=None,
    use_box_filter: bool = False,
):
    assert (source_latent.device == target_latent.device)
    assert (source_latent.shape == target_latent.shape)
    B, C, H, W = source_latent.shape

    # 2D FFT
    source_fft_shift = get_spectrum(source_latent.to(torch.float32))
    target_fft_shift = get_spectrum(target_latent.to(torch.float32))

    # magnitude & phase of 2D FFT
    source_magnitude = source_fft_shift.abs()
    source_phase = torch.angle(source_fft_shift)
    target_phase = torch.angle(target_fft_shift)

    # gaussian sigma setting
    init_lambda, min_lambda = gaussian_lambda, 0
    curr_lambda = gaussian_lambda
    
    if decay_type == None:
        # SNR-based decay (ours)
        pass
    elif decay_type == "convex":
        # exponential decay
        progress = step_index / max(1, total_timesteps - 1)
        decay_rate = 3.0
        curr_lambda = min_lambda + (init_lambda - min_lambda) * np.exp(-decay_rate * progress)
    elif decay_type == "concave":
        # inverse exponential decay (logarithmetic-like)
        progress = step_index / max(1, total_timesteps - 1)
        decay_rate = 3.0
        curr_lambda = min_lambda + (init_lambda - min_lambda) * (1.0 - np.exp(-decay_rate * (1 - progress)))
    elif decay_type == "step":
        # no setting (all steps are equal)
        if step_index >= 24: # 60% for swapping, 40% for non-swapping (at num_ddim_inversion=50)
            curr_lambda = 0.0

    # make (gaussian) low-pass filter
    if use_box_filter == True:
        mask = make_box_lowpass_filter(H, W, source_latent.device, curr_lambda)[None, None, :, :]
    else:
        mask = make_gaussian_lowpass_filter(H, W, source_latent.device, curr_lambda)[None, None, :, :]
    mask = mask * phase_strength
    mask = mask.clamp(0.0, 1.0)

    ## low-frequency phase refinement (vector interpolation)
    source_phase_cos = torch.cos(source_phase)
    source_phase_sin = torch.sin(source_phase)
    target_phase_cos = torch.cos(target_phase)
    target_phase_sin = torch.sin(target_phase)
    mixed_real = source_phase_cos * (1.0 - mask) + target_phase_cos * mask
    mixed_imag = source_phase_sin * (1.0 - mask) + target_phase_sin * mask

    mixed_phase = torch.atan2(mixed_imag, mixed_real)
    swapped_fft_shift = torch.polar(source_magnitude, mixed_phase)

    swapped_fft = torch.fft.ifftshift(swapped_fft_shift, dim=(-2, -1))
    swapped_latent = torch.fft.ifft2(swapped_fft, dim=(-2, -1)).real

    return swapped_latent