import argparse
import os
import torch

from diffusers import (
    StableDiffusionXLImg2ImgPipeline,
    DDIMInverseScheduler,
    DDIMScheduler, 
    DPMSolverMultistepScheduler
)

from pytorch_lightning import seed_everything
from functools import partial
from PIL import Image

from modules.ddim import ddim_inversion
from modules.fourier import fourier_swapping


def callback_dynamic_latents(
    pipe,
    step_index,
    timestep,
    callback_kwargs, 
    ddim_latents=None,
    init_gaussian_lambda=0.3,
    num_inference_steps=50,
    strength=0.8,
    step_offset=0,
):
    latents = callback_kwargs["latents"]

    t = timestep.item()
    idx = int(round(t / 1000.0) * num_inference_steps)
    idx = min(idx, len(ddim_latents) - 1)
    idx = max(0, idx)
    total_swapping_timesteps = int(num_inference_steps * strength)

    current_step_index = step_index + step_offset

    target_latent = ddim_latents[idx]
    target_latent = target_latent.to(latents.device, dtype=latents.dtype)

    # SNR-based decay scheduling
    t_idx = int(t)
    alphas_cumprod = pipe.scheduler.alphas_cumprod.to(latents.device)
    current_alpha = alphas_cumprod[t_idx]
    decay_factor = 1.0 - current_alpha.item()
    gaussian_lambda = init_gaussian_lambda * decay_factor

    swapped_latent = fourier_swapping(
        source_latent=latents,
        target_latent=target_latent,
        step_index=current_step_index,
        total_timesteps=total_swapping_timesteps,
        gaussian_lambda=gaussian_lambda,
    )
    swapped_latent = swapped_latent.to(dtype=latents.dtype)
    callback_kwargs["latents"] = swapped_latent

    return callback_kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-image", type=str, default="./images/content_images/qual1.png") # TODO
    parser.add_argument("--prompt", type=str, default="")                                         # TODO
    parser.add_argument("--output-dir", type=str, default="./images/stylized_images")             # TODO
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--denoising-sampler", type=str, default="Euler", choices=["Euler", "DDIM", "DPM++2M"])
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--sdxl-base-ratio", type=float, default=0.8)
    parser.add_argument("--model-cache-dir", type=str, default="./model_cache") # TODO
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load SDXL model (Base and Refiner)
    print(f"Loading SDXL Base model...")
    pipe_base = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=args.model_cache_dir,
    ).to(device)
    pipe_base.vae.to(dtype=torch.float32)

    print(f"Loading SDXL Refiner model...")
    pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=args.model_cache_dir,
    ).to(device)

    # configure denoising scheduler
    if args.denoising_sampler != "Euler":
        if args.denoising_sampler == "DDIM":
            pipe_base.scheduler = DDIMScheduler.from_config(pipe_base.scheduler.config)
            pipe_refiner.scheduler = DDIMScheduler.from_config(pipe_refiner.scheduler.config)
        elif args.denoising_sampler == "DPM++2M":
            pipe_base.scheduler = DPMSolverMultistepScheduler.from_config(pipe_base.scheduler.config)
            pipe_refiner.scheduler = DPMSolverMultistepScheduler.from_config(pipe_refiner.scheduler.config)

    # load and preprocess content image
    init_image = Image.open(args.content_image).convert("RGB")
    init_image = init_image.resize((1024, 1024))

    # perform DDIM inversion
    print(f"Running DDIM inverion for {args.num_inference_steps} steps...")
    ddim_latents = ddim_inversion(
        pipe=pipe_base,
        init_image=init_image,
        prompt="",
        num_ddim_steps=args.num_inference_steps,
        guidance_scale=1.0,
    )

    # configure callback functions
    callback_base_fn = partial(
        callback_dynamic_latents,
        ddim_latents=ddim_latents,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        step_offset=0,
    )
    callback_refiner_fn = partial(
        callback_dynamic_latents,
        ddim_latents=ddim_latents,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        step_offset=30, # the actual number of steps that Base will perform
    )

    # run base denoising
    print("Running base pipeline...")
    latents = pipe_base(
        image=init_image,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        denoising_end=args.sdxl_base_ratio,
        output_type="latent",
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        callback_on_step_end=callback_base_fn,
        callback_on_step_end_tensor_inputs=["latents"]
    ).images

    # run refiner denoising
    print("Running refiner pipeline...")
    image = pipe_refiner(
        image=latents,
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        denoising_start=args.sdxl_base_ratio,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        callback_on_step_end=callback_refiner_fn,
        callback_on_step_end_tensor_inputs=["latents"]
    ).images[0]

    # save stylized image
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    content_filename = os.path.splitext(os.path.basename(args.content_image))[0]
    save_filename = f"{content_filename}_{args.prompt}.png"
    save_path = os.path.join(args.output_dir, save_filename)
    image.save(save_path)
    print(f"Stylized image saved to {save_path}")


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=9 \
python adaphase_sdxl.py \
  --content-image <your content image direction.png> \
  --prompt <your stylization prompt> \
  --output-dir <your stylized image output direction> \
  --model-cache-dir "/data/lfs/sekwang/sdxl-parameters/"

Example style keyword:
    "mosaic art"
    "oil painting"
    "claymation"
    "cartoon"
    "voxel art"
    "sketching"
    "pixel art"
    "ukiyo-e"
    "watercolor"
    "cubism"
    "cyberpunk"
    "pointillism"
    "ink wash painting"
    "impasto painting"
    "stained glass art"
    "glitch art"
    "graffiti panting"
    "origami"
    "impressionism"
    "low poly"
"""