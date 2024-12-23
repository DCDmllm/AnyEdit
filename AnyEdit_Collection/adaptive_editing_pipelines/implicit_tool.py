# ref to https://github.com/YangLing0818/EditWorld/tree/main/t2i_branch_base.py
import argparse
import os
import re
import json
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    DDIMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionControlNetInpaintPipeline,
    UniPCMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    AutoencoderKL
    )
from PIL import Image
import numpy as np
import random
import cv2

from AnyEdit_Collection.other_modules.ip_adapter import IPAdapterXL
from AnyEdit_Collection.other_modules.prompt2prompt.prompt_to_prompt_stable import mask_from_CA, AttentionStore, run_and_display
import clip
device_0='cuda:0'
device_1='cuda:1'

## The basic judgment version is to use the CLIP score
def CLIP_score(img, text, device='cuda', jit=False):
    """
    img: PIL
    """
    model, preprocess = clip.load(name="ViT-L/14", device=device, jit=jit)
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
    similarity = F.cosine_similarity(image_features, text_features)
    return similarity


def CLIP_score_I(img1, img2, device='cuda', jit=False):
    """
    img: PIL
    """
    model, preprocess = clip.load(name="ViT-L/14", device=device, jit=jit)
    image1 = preprocess(img1).unsqueeze(0).to(device)
    image2 = preprocess(img2).unsqueeze(0).to(device)
    with torch.no_grad():
        image1_features = model.encode_image(image1)
        image2_features = model.encode_image(image2)
    similarity = F.cosine_similarity(image1_features, image2_features)
    return similarity


def compute_structure_comparison(img1, img2, C3=1e-3):
    """
    img1, img2: PIL
    """
    sigma1_sq = np.var(np.array(img1))
    sigma2_sq = np.var(np.array(img2))

    sigma12 = np.cov(np.array(img1).flatten(), np.array(img2).flatten())[0, 1]

    structure_comparison = (sigma12 + C3) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C3)
    
    return structure_comparison


def obtain_stage1_image(ori_text, key_words, controller, ori_gen_time=1):
    best_clip_score = -999
    for _ in range(ori_gen_time):
        # ori_img_ = pipe_text2img(ori_text).images[0]
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        ori_img_, _ = run_and_display([ori_text], controller, latent=None, run_baseline=False, generator=g_cpu)
        ori_mask_ = mask_from_CA(controller, res=16, from_where=("up", "down"), prompts=[ori_text], key_words=key_words)
        clip_score = CLIP_score(ori_img_, ori_text, device=device_0)
        if best_clip_score < clip_score:
            best_clip_score = clip_score
            ori_img = ori_img_
            ori_mask = ori_mask_
    # print("best_clip_score for [Stage 1]: ", best_clip_score)
    return ori_img, ori_mask


def obtain_inpainting_results(ori_text, tar_text, ori_img, ori_mask, gen_time_ori=1, gen_time_tar=1):
    best_clip_score, SSIM = -999, -999
    for _ in range(gen_time_ori):
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        ori_img_ = pipe_inpaint(prompt=ori_text, image=ori_img, mask_image=ori_mask.resize((512, 512))).images[0]
        clip_score = CLIP_score(ori_img_, ori_text, device=device_0)
        if best_clip_score < clip_score:
            best_clip_score = clip_score
            ori_img_1 = ori_img_
    # print("best_clip_score for [Stage 1]: ", best_clip_score)
    for _ in range(gen_time_tar):
        g_cpu = torch.Generator().manual_seed(random.randint(0, 888888))
        tar_img_ = pipe_inpaint(prompt=tar_text, image=ori_img, mask_image=ori_mask.resize((512, 512))).images[0]
        ssim = CLIP_score(tar_img_, tar_text, device=device_0) + \
               CLIP_score_I(tar_img_, ori_img_1, device=device_0) + compute_structure_comparison(tar_img_, ori_img_1)
        if SSIM < ssim:
            SSIM = ssim
            tar_img_1 = tar_img_
    print("[Stage 1] SSIM : ", SSIM)
    return ori_img_1, tar_img_1


def obtain_ip_tar_imgs(ip_model, ori_img, tar_img, tar_text, tar_img_canny, ori_mask, gen_time=1):
    # best_clip_I = CLIP_score_I(img1=tar_img, img2=ori_img) + 4*CLIP_score(tar_img, tar_text)
    # tar_img_1 = tar_img
    best_clip_I = -999
    for _ in range(gen_time):
        tar_img_ = ip_model.generate(pil_image=ori_img, prompt=tar_text, num_samples=1, num_inference_steps=50, seed=42,
                                image=tar_img, mask_image=ori_mask.resize(tar_img.size), control_image=tar_img_canny)[0]
        clip_I = CLIP_score_I(img1=tar_img_, img2=ori_img, device=device_0) + \
                 4*CLIP_score(tar_img_, tar_text, device=device_0)
        if best_clip_I < clip_I:
            tar_img_1 = tar_img_
            best_clip_I = clip_I
    return tar_img_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model-path', type=str,
                        default='./checkpoints/foundation_models/stable-diffusion-xl-base-1.0')
    parser.add_argument('--image-encoder-path', type=str,
                        default='./checkpoints/visual_models/ip_adapter/sdxl_models/image_encoder')
    parser.add_argument('--ip-ckpt', type=str,
                        default='./checkpoints/visual_models/ip_adapter/sdxl_models/ip-adapter_sdxl.bin')
    parser.add_argument('--controlnetXL-ckpt', type=str,
                        default='./checkpoints/visual_models/controlnet-canny-sdxl-1.0')
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    opt = parser.parse_args()

    # load ckpt
    base_model_path = opt.base_model_path
    image_encoder_path = opt.image_encoder_path
    ip_ckpt = opt.ip_ckpt
    controlnetXL_ckpt = opt.controlnetXL_ckpt
    ## load SDXL pipeline
    pipe_inpaint = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
        add_watermarker=False,
        use_safetensors=True
    ).to(device_0)
    pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        base_model_path,
        variant="fp16",
        torch_dtype=torch.float16,
        add_watermarker=False,
        use_safetensors=True
    ).to(device_0)

    ## load controlnet and ipadapter
    controlnet = ControlNetModel.from_pretrained(
        controlnetXL_ckpt,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe_control_xl = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        add_watermarker=False,
        use_safetensors=True
    ).to(device_1)
    ip_model = IPAdapterXL(pipe_control_xl, image_encoder_path, ip_ckpt, device_1)


    
    with open(opt.json_path, 'r') as f:
        texts_dict = json.load(f)
    os.makedirs(os.path.join(opt.instruction_path, opt.instruction_type, f'input_img'), exist_ok=True)
    os.makedirs(os.path.join(opt.instruction_path, opt.instruction_type, f'edited_img'), exist_ok=True)
    output_results = []
    for key in texts_dict.keys():
        sample_id = int(key.replace("sample", ""))
        group_id = sample_id // 100
        index = sample_id % 100

        # textual prompt
        ori_text = re.sub(r'[^\w\s]', '', texts_dict[key]["original_caption"])
        tar_text = texts_dict[key]["target_caption"]
        key_words = texts_dict[key]["key_words"].split(", ")
        for i in range(len(key_words)):
            wd = key_words[i]
            wd = re.sub(r'[^\w\s]', '', wd)
            if (' ' in wd) and (wd in ori_text):
                wds = wd.split(' ')
                key_words[i] = wds[0]

        # <<<< Stage 1： generate image >>>>
        controller = AttentionStore()
        ori_img, ori_mask = obtain_stage1_image(ori_text, key_words, controller, ori_gen_time=1)

        with torch.no_grad():
            # inpainting
            ori_img, tar_img = obtain_inpainting_results(ori_text=ori_text, tar_text=tar_text, ori_img=ori_img,
                                                         ori_mask=ori_mask, gen_time_ori=1, gen_time_tar=1) 
            ori_img = pipe_img2img(prompt=ori_text + ", realistic", image=ori_img, strength=0.5).images[0]

            # <<<< Stage 2： optimize Figure 2 >>>>
            # obtain canny of tar_img
            tar_img_canny = np.array(tar_img)
            tar_img_canny = cv2.Canny(tar_img_canny, 100, 200)
            tar_img_canny = tar_img_canny[:, :, None]
            tar_img_canny = np.concatenate([tar_img_canny, tar_img_canny, tar_img_canny], axis=2)
            tar_img_canny = Image.fromarray(tar_img_canny)
            # ip-adapter generation
            tar_img = obtain_ip_tar_imgs(ip_model=ip_model, ori_img=ori_img, tar_img=tar_img, tar_text=tar_text,
                                         tar_img_canny=tar_img_canny, ori_mask=ori_mask, gen_time=1)
            tar_img = pipe_img2img(prompt=tar_text + ", realistic", image=tar_img, strength=0.5).images[0]
            print("[Stage 2] end")
        ori_img_path = os.path.join(opt.instruction_path, 'input_img', f"group_{group_id}_sample_{index}_ori.jpg")
        tar_img_path = os.path.join(opt.instruction_path, 'edited_img', f"group_{group_id}_sample_{index}_tar.jpg")
        ori_img.save(ori_img_path)
        tar_img.save(tar_img_path)
        cur_data = {"edit": texts_dict[key]["instruction"], "output": tar_text, "input": ori_text, "edit_type": opt.instruction_type, "image_file": ori_img_path, "edited_file": tar_img_path}
        output_results.append(cur_data)
    print(f"valid editing insturction data: {len(output_results)}")
    with open(f'{opt.instruction_path}/{opt.instruction_type}/final_edit_results_-1_-1.json', 'w') as results_file:
        json.dump(output_results, results_file, indent=4)
