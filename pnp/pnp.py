import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline
import re
from pnp_utils import *
import numpy as np

# suppress partial model loading warning
logging.set_verbosity_error()


class PNP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        model_key = "lambdalabs/sd-pokemon-diffusers"
        self.opt = opt 
        self.image = None
        self.eps = None

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(50, device='cuda')
        print('SD model loaded')

        # Load image and initialize embeddings
        self.load_data()
        self.text_embeds = self.get_text_embeds('a cartoon of pokemon', 'realistic, blurry, low res')
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to('cuda'))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to('cuda'))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def load_data(self):
        image_path = self.opt.data_path
        
        if os.path.isdir(image_path):
            file_list = os.listdir(image_path)
            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
            sorted_files = sorted(file_list, key=natural_sort_key)
        else:
            sorted_files = [image_path]

        img_idx = 0
        for img in tqdm(sorted_files):
            if os.path.isdir(image_path):
                img_fullpath = os.path.join(image_path, img)
            else:
                img_fullpath = image_path
            file_name_with_ext = os.path.basename(img_fullpath)
            img_name = os.path.splitext(file_name_with_ext)[0]
            
            if img_idx < self.opt.start_index:
                print(f"{img_name} skipping!")
                img_idx += 1
                continue
            image = Image.open(os.path.join(image_path, img)).convert('RGB')
            image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
            image = T.ToTensor()(image).to('cuda')
            latents_path = os.path.join('/home/aikusrv04/pokemon/similar_pokemon/pnp/latents_forward/', img_name, f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
            noisy_latent = torch.load(latents_path).to('cuda')
            yield image, noisy_latent, img_name

    @torch.no_grad()
    def denoise_step(self, x, t):
        register_time(self, t.item()) 
        latent_model_input = torch.cat([self.eps] + [x] * 2)
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)

        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self):
        for image, noisy_latent, img_name in self.load_data():
            self.image = image
            self.eps = noisy_latent
            self.img_name = img_name

            print(f"Processing {self.img_name}...")
            self.sample_loop(self.eps)

    def sample_loop(self, x):
        img_name = f"{self.img_name}.png"  # 출력 이미지 이름
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t)
            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(os.path.join(self.opt.save_dir, img_name))
        return decoded_latent



if __name__ == '__main__':
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='output')
    parser.add_argument('--start_index', type=int, default=0)
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)
    seed_everything(1)

    pnp = PNP(opt)
    pnp.run_pnp()

