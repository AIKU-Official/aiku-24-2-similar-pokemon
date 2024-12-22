from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import os
from PIL import Image
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from pathlib import Path
from pnp_utils import *
import torchvision.transforms as T
import glob
import re


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class Preprocess(nn.Module):
    def __init__(self, device, hf_key=None):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        model_key = "lambdalabs/sd-pokemon-diffusers"    

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", revision="main",
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="main",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="main",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')

        self.inversion_func = self.ddim_inversion

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path,device):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, save_path, save_latents=True,
                                timesteps_to_save=None):
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                if save_latents:
                    torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        torch.save(latent, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return latent

    @torch.no_grad()
    def ddim_sample(self, x, cond, save_path, save_latents=False, timesteps_to_save=None):
        timesteps = self.scheduler.timesteps
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(timesteps)):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps

            if save_latents:
                torch.save(x, os.path.join(save_path, f'noisy_latents_{t}.pt'))
        return x

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path, save_path, timesteps_to_save,
                        inversion_prompt='human', extract_reverse=False, device='cuda'):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path,device)
        latent = self.encode_imgs(image)

        inverted_x = self.inversion_func(cond, latent, save_path, save_latents=not extract_reverse,
                                         timesteps_to_save=timesteps_to_save)
        latent_reconstruction = self.ddim_sample(inverted_x, cond, save_path, save_latents=extract_reverse,
                                                 timesteps_to_save=timesteps_to_save)
        rgb_reconstruction = self.decode_latents(latent_reconstruction)

        return rgb_reconstruction  # , latent_reconstruction


# def run(opt):
#     # timesteps to save

#     model_key = "lambdalabs/sd-pokemon-diffusers"       
#     toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
#     toy_scheduler.set_timesteps(opt.save_steps)
#     timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.save_steps,
#                                                            strength=1.0,
#                                                            device=device)

#     seed_everything(1)

#     extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
#     save_path = os.path.join(opt.save_dir + extraction_path_prefix, os.path.splitext(os.path.basename(opt.data_path))[0])
#     os.makedirs(save_path, exist_ok=True)

#     model = Preprocess(device, hf_key=None)
#     recon_image = model.extract_latents(data_path=opt.data_path,
#                                          num_steps=opt.steps,
#                                          save_path=save_path,
#                                          timesteps_to_save=timesteps_to_save,
#                                          inversion_prompt='human',
#                                          extract_reverse=opt.extract_reverse)






###############병렬적으로 실행할 때#################3
# def run(opt):
#     # timesteps to save
#     model_key = "lambdalabs/sd-pokemon-diffusers"       
#     toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
#     toy_scheduler.set_timesteps(opt.save_steps)
#     timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=opt.save_steps,
#                                                            strength=1.0,
#                                                            device=device)

#     seed_everything(1)

#     model = Preprocess(device, hf_key=None)

#     # # Check if data_path is a directory
#     # if os.path.isdir(opt.data_path):
#     #     image_paths = glob.glob(os.path.join(opt.data_path, "*.jpg")) + glob.glob(os.path.join(opt.data_path, "*.png"))
#     # else:
#     #     image_paths = [opt.data_path]

#     image_path=opt.data_path
#     file_list = os.listdir(image_path)
    
    
#     def natural_sort_key(s):
#         return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
#     sorted_files = sorted(file_list, key=natural_sort_key)
#     # file_path = f"{output_dir}/output.txt"

#     # output_lst = []
#     # with open(file_path, 'r', encoding='utf-8') as file:  # utf-8 인코딩을 설정합니다
#     #     for line in file:
#     #         output_lst.append(line.strip())
    
#     img_idx=0
#     for img in tqdm(sorted_files):
#         img_name = os.path.splitext(img)[0]
#         # output_img = f"{output_dir}/{img_name}.png"
#         # if (os.path.exists(output_img)) or (img_idx < opt.start_index) or (img in output_lst):
#         if (img_idx < opt.start_index):
#             print(f"{img_name} skipping!")
#             img_idx += 1
#             continue
        
#         # image_pil, image = load_image(os.path.join(image_path,img))


#     # for image_path in tqdm(image_paths, desc="Processing Images"):
#         extraction_path_prefix = "_reverse" if opt.extract_reverse else "_forward"
#         save_path = os.path.join(opt.save_dir + extraction_path_prefix, img_name)
#         os.makedirs(save_path, exist_ok=True)

#         recon_image = model.extract_latents(data_path=str(os.path.join(image_path, img)),
#                                             num_steps=opt.steps,
#                                             save_path=save_path,
#                                             timesteps_to_save=timesteps_to_save,
#                                             inversion_prompt='human',
#                                             extract_reverse=opt.extract_reverse)

#     # T.ToPILImage()(recon_image[0]).save(os.path.join(save_path, f'recon.jpg'))



# if __name__ == "__main__":
#     device = 'cuda'
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', type=str,
#                         default='data/horse.jpg')
#     parser.add_argument('--save_dir', type=str, default='latents')
#     parser.add_argument('--steps', type=int, default=999)
#     parser.add_argument('--save-steps', type=int, default=1000)
#     parser.add_argument('--extract-reverse', default=False, action='store_true', help="extract features during the denoising process")
#     parser.add_argument('--start_index', type=int, default=0)
#     opt = parser.parse_args()
#     run(opt)


##############retrieval이랑 같이 실행할 때 #############3
def run_preprocess(data_path, save_dir, steps=999, save_steps=1000, extract_reverse=False, start_index=0, device="cuda"):
    """
    Preprocess.py에서 실행 부분을 함수화한 코드.
    
    Args:
        data_path (str): 입력 데이터 경로 (파일 또는 디렉토리).
        save_dir (str): 출력 디렉토리 경로.
        steps (int): 디노이징 과정의 단계 수.
        save_steps (int): 저장 간격.
        extract_reverse (bool): 역 디노이징 과정 여부.
        start_index (int): 처리 시작 인덱스.
        device (str): 처리에 사용할 장치 ("cuda" 또는 "cpu").
    """
    # timesteps를 설정
    model_key = "lambdalabs/sd-pokemon-diffusers"       
    toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    toy_scheduler.set_timesteps(save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=save_steps,
                                                           strength=1.0, device=device)

    # 재현성 설정
    seed_everything(1)

    # 모델 초기화
    model = Preprocess(device, hf_key=None)

    # 이미지 경로 처리
    if os.path.isdir(data_path):
        file_list = os.listdir(data_path)
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
        sorted_files = sorted(file_list, key=natural_sort_key)
    else:
        sorted_files = [data_path]

    img_idx = 0
    for img in tqdm(sorted_files, desc="Processing Images"):
        img_name = os.path.splitext(os.path.basename(img))[0]
        if img_idx < start_index:
            print(f"{img_name} skipping!")
            img_idx += 1
            continue

        # 출력 경로 설정
        extraction_path_prefix = "_reverse" if extract_reverse else "_forward"
        save_path = os.path.join(save_dir + extraction_path_prefix, img_name)
        os.makedirs(save_path, exist_ok=True)

        # 라텐트 추출
        recon_image = model.extract_latents(data_path=os.path.join(data_path, img),
                                            num_steps=steps,
                                            save_path=save_path,
                                            timesteps_to_save=timesteps_to_save,
                                            inversion_prompt='human',
                                            extract_reverse=extract_reverse)
        print(f"Processed {img_name}, saved to {save_path}")


if __name__ == "__main__":
    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/home/aikusrv04/pokemon/similar-pokemon/example", help="Input data path (image or directory).")
    parser.add_argument('--save_dir', type=str, default="/home/aikusrv04/pokemon/similar-pokemon/pnp/output/preprocess", help="Directory to save the outputs.")
    parser.add_argument('--steps', type=int, default=999, help="Number of denoising steps.")
    parser.add_argument('--save_steps', type=int, default=1000, help="Timesteps for saving.")
    parser.add_argument('--extract_reverse', default=False, action='store_true', help="Extract features during the denoising process.")
    parser.add_argument('--start_index', type=int, default=0, help="Start index for processing images.")
    opt = parser.parse_args()

    run_preprocess(data_path=opt.data_path, 
                   save_dir=opt.save_dir, 
                   steps=opt.steps, 
                   save_steps=opt.save_steps, 
                   extract_reverse=opt.extract_reverse, 
                   start_index=opt.start_index, 
                   device=device)
