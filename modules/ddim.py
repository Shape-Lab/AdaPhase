import torch
from diffusers import DDIMInverseScheduler


@torch.no_grad()
def ddim_inversion(pipe, init_image, prompt="", num_ddim_steps=50, guidance_scale=1.0):
    original_scheduler = pipe.scheduler
    scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler

    image = pipe.image_processor.preprocess(init_image).to(pipe.device, dtype=pipe.vae.dtype)

    # VAE encode
    latents = pipe.vae.encode(image).latent_dist.mode()
    latents = latents * pipe.vae.config.scaling_factor
    latents = latents.to(dtype=pipe.dtype)

    # prompt embedding
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) \
        = pipe.encode_prompt(
            prompt=prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=(guidance_scale > 1.0)
        )

    # set timesteps
    pipe.scheduler.set_timesteps(num_ddim_steps, device=pipe.device)
    timesteps = pipe.scheduler.timesteps

    ddim_latents = [latents.clone()]

    for t in timesteps:
        if guidance_scale > 1.0:
            latent_model_input = torch.cat([latents] * 2)
            current_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            current_pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        else:
            latent_model_input = latents
            current_prompt_embeds = prompt_embeds
            current_pooled = pooled_prompt_embeds

        # generate time_ids
        height, width = latents.shape[-2:]
        height, width = height * 8, width * 8
        original_size, target_size = (height, width), (height, width)
        crops_coords_top_left = (0, 0)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=pipe.dtype, device=pipe.device)
        if guidance_scale > 1.0:
            add_time_ids = torch.cat([add_time_ids] * 2)

        added_cond_kwargs = {
            "text_embeds": current_pooled,
            "time_ids": add_time_ids,
        }

        # predict noise
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=current_prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # DDIM inverison step
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        ddim_latents.append(latents.clone())

    # restore original scheduler
    pipe.scheduler = original_scheduler

    return ddim_latents