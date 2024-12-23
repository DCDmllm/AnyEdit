# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
from rich.console import Console
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType
from torch.cuda.amp import custom_bwd, custom_fwd
import torchvision

CONSOLE = Console(width=120)

# try:
from diffusers import (
    DDIMScheduler,
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
from transformers import logging

# except ImportError:
#     CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
#     CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
#     CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
#     sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, model_path, scheduler_path, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None, local_files_only=True)
        pipe.scheduler = DDIMScheduler.from_pretrained(scheduler_path, subfolder="scheduler", local_files_only=True)
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            pipe.unet.float()
            pipe.vae.float()
        else:
            if self.device.index:
                pipe.enable_model_cpu_offload(self.device.index)
            else:
                pipe.enable_model_cpu_offload(0)

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae
        CONSOLE.print("InstructPix2Pix loaded!")

    def edit_image(
        self,
        cond_embedding: TensorType["N", "max_length", "embed_dim"],
        uncond_embedding: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        mask_threshold=0.5,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        wys_noise_level: float = 0.8,
        mask=None,
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """
        image, image_cond = (image + 1) / 2, (image_cond + 1) / 2

        min_step = int(self.num_train_timesteps * 1.0)
        max_step = int(self.num_train_timesteps * 1.0)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(latents)

        original_latents = self.imgs_to_latent(image_cond)

        if mask is not None:
            heatmap_hq, heatmap = self.prepare_mask_and_mask_latents(mask, image.size()[2], image.size()[3])
            mask_latent = heatmap.float().to(self.device)
            mask = heatmap_hq.float().to(self.device)
        else:
            heatmap, heatmap_hq = self.predict_mask(original_latents, cond_embedding, uncond_embedding, image, noise_level=wys_noise_level)
            mask_latent = (heatmap >= mask_threshold).float().to(self.device)
            mask = (heatmap_hq >= mask_threshold).float().to(self.device)
        text_embedding = torch.cat([cond_embedding, uncond_embedding, uncond_embedding], dim=0)
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore

        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                outputs = self.unet(latent_model_input, t, encoder_hidden_states=text_embedding).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = outputs.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # replacing the unmasked latents
            tmp = self.scheduler.add_noise(original_latents, noise, self.scheduler.timesteps[i])	
            latents = latents * mask_latent + tmp * (1 - mask_latent)

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)
        decoded_img = torchvision.transforms.Resize(image_cond.shape[2:])(decoded_img)
        decoded_img = decoded_img * mask + image_cond * (1 - mask)

        return decoded_img * 2 - 1, heatmap_hq

    @torch.no_grad()
    def get_noise_diff(self, noise_cond, noise_uncond):
        diff = (noise_cond - noise_uncond).abs()[0].sum(dim=0).detach().cpu().numpy()

        # removing outliers
        Q1 = np.percentile(diff, 25, interpolation = 'midpoint') 
        Q3 = np.percentile(diff, 75, interpolation = 'midpoint') 
        IQR = Q3 - Q1
        factor = 1.5
        low_lim = Q1 - factor * IQR
        up_lim = Q3 + factor * IQR
        diff = np.clip(diff, 0, up_lim)

        # normalizing to [0, 1]
        diff = (diff - diff.min()) / (diff.max() - diff.min())

        return diff

    @torch.no_grad()
    def get_noise_preds(self, latents, image_cond_latents, text_embedding, noise_level):
        t = torch.tensor([int(noise_level * self.num_train_timesteps)])
        noisy_latents = self.scheduler.add_noise(latents, torch.randn_like(latents), t)
        latent_model_input = torch.cat([noisy_latents] * 3)
        latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
        outputs = self.unet(latent_model_input, t.item(), encoder_hidden_states=text_embedding).sample
        noise_pred_text, noise_pred_image, noise_pred_uncond = outputs.chunk(3)
        return noise_pred_text, noise_pred_image, noise_pred_uncond

    @torch.no_grad()
    def predict_mask(self, latents, cond_embedding, uncond_embedding, image, noise_level):
        image_cond_latents = self.prepare_image_latents(image)
        text_embedding = torch.cat([cond_embedding, uncond_embedding, uncond_embedding], dim=0)

        noise_pred_text, noise_pred_image, noise_pred_uncond = self.get_noise_preds(latents, image_cond_latents, text_embedding, noise_level)

        diff = self.get_noise_diff(noise_pred_text, noise_pred_image)  # diff is in the latent space resolution, of shape ("H//8", "W//8")
        heatmap = torch.from_numpy(diff[None, None, ...])
    
        heatmap_hq = F.interpolate(heatmap, size=image.size()[2:], mode="bilinear")

        return heatmap, heatmap_hq
    
    def prepare_mask_and_mask_latents(self, mask, height, width):
        
        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask).float()
        
        mask_latents = torch.nn.functional.interpolate(
            mask, size=(height // 8, width // 8), mode="bilinear"
        )
        # mask_latents = mask_latents.to(device=device, dtype=dtype)

        return mask, mask_latents
    
    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latent = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latent], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    @torch.no_grad()
    def ddim_loop(self, latent, text_embedding: TensorType, diffusion_steps: int = 20):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(diffusion_steps):
            t = self.pipe.scheduler.timesteps[len(self.pipe.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, text_embedding)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad() 
    def ddim_inversion(self, latents, text_embedding: TensorType, diffusion_steps: int = 20):
        latents_ddim = self.ddim_loop(latents, text_embedding, diffusion_steps=diffusion_steps)
        return latents_ddim

    def get_noise_pred_single(self, latents, t, context, 
                              guidance_scale: float = 7.5,
                              image_guidance_scale: float = 1.5):
        latent_model_input = torch.cat([latents] * 3)
        latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=1)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=context)["sample"]
        
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )
        
        return noise_pred

class LocalEditor(nn.Module):
    def __init__(self, model_path, scheduler_path):
        super().__init__()
        
        self.device = (torch.device('cuda'))
        self.ip2p = InstructPix2Pix(model_path, scheduler_path, self.device, ip2p_use_full_precision=True)
        print("Loaded WYS-IP2P")

    def forward(
        self,
        image: torch.Tensor,
        edit: str,
        scale_txt: float = 7.5,
        scale_img: float = 1.0,
        steps: int = 100,
        mask_threshold=0.5,
        return_heatmap=False,
        check_size=True,
        mask=None,
    ) -> torch.Tensor:
        assert image.dim() == 3
        if check_size:
            assert image.size(1) % 64 == 0
            assert image.size(2) % 64 == 0
        with torch.no_grad():
            text_embedding = self.ip2p.pipe._encode_prompt(
                edit, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
            )
            cond_embedding, uncond_embedding, _ = text_embedding.chunk(3)

        with torch.no_grad(), torch.autocast("cuda"):
            edited_image, heatmap = self.ip2p.edit_image(
                cond_embedding.float().to(self.device),
                uncond_embedding.float().to(self.device),
                image[None, ...].float().to(self.device),
                image[None, ...].float().to(self.device),
                guidance_scale=scale_txt,
                image_guidance_scale=scale_img,
                diffusion_steps=steps,
                mask_threshold=mask_threshold,
                mask=mask,
            )
            if return_heatmap:
                return edited_image[0], heatmap
            return edited_image[0]