import os
import uuid
import torch
import numpy as np
from transformers.models.phi3.modeling_phi3 import Phi3PreTrainedModel
import random
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from omost.lib_omost.pipeline import StableDiffusionXLOmostPipeline
import omost.lib_omost.canvas as omost_canvas
import omost.lib_omost.memory_management as memory_management
from termcolor import cprint

os.environ["PYTHONPATH"] = './'

Phi3PreTrainedModel._supports_sdpa = True

sdxl_name = './checkpoints/foundation_models/RealVisXL_V4' # sdxl_name = 'stabilityai/stable-diffusion-xl-base-1.0'
llm_name = './checkpoints/foundation_models/omost-llama-3-8b-4bits'

tokenizer = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(
    sdxl_name, subfolder="tokenizer_2")
text_encoder = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16")
text_encoder_2 = CLIPTextModel.from_pretrained(
    sdxl_name, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16")
vae = AutoencoderKL.from_pretrained(
    sdxl_name, subfolder="vae", torch_dtype=torch.bfloat16, variant="fp16")  # bfloat16 vae
unet = UNet2DConditionModel.from_pretrained(
    sdxl_name, subfolder="unet", torch_dtype=torch.float16, variant="fp16")

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

pipeline = StableDiffusionXLOmostPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    unet=unet,
    scheduler=None,  # We completely give up diffusers sampling system and use A1111's method
)

memory_management.unload_all_models([text_encoder, text_encoder_2, vae, unet])

llm_model = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

llm_tokenizer = AutoTokenizer.from_pretrained(
    llm_name,
)
memory_management.unload_all_models(llm_model)

@torch.inference_mode()
def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((target_width, target_height), Image.LANCZOS)
    return np.array(resized_image)


@torch.inference_mode()
def post_chat(history):
    canvas_outputs = None

    try:
        if history:
            history = [(user, assistant) for user, assistant in history if isinstance(user, str) and isinstance(assistant, str)]
            last_assistant = history[-1][1] if len(history) > 0 else None
            canvas = omost_canvas.Canvas.from_bot_response(last_assistant)
            canvas_outputs = canvas.process()
    except Exception as e:
        print('Last assistant response is not valid canvas:', e)

    return canvas_outputs


@torch.inference_mode()
def diffusion_fn(canvas_outputs, num_samples, seed, image_width, image_height,
                 highres_scale, steps, cfg, highres_steps, highres_denoise, negative_prompt, save_dir, unique_hex, relation):

    use_initial_latent = False
    eps = 0.05

    image_width, image_height = int(image_width // 64) * 64, int(image_height // 64) * 64

    rng = torch.Generator(device=memory_management.gpu).manual_seed(seed)

    memory_management.load_models_to_gpu([text_encoder, text_encoder_2])

    positive_cond, positive_pooler, negative_cond, negative_pooler = pipeline.all_conds_from_canvas(canvas_outputs, negative_prompt)

    if use_initial_latent:
        memory_management.load_models_to_gpu([vae])
        initial_latent = torch.from_numpy(canvas_outputs['initial_latent'])[None].movedim(-1, 1) / 127.5 - 1.0
        initial_latent_blur = 40
        initial_latent = torch.nn.functional.avg_pool2d(
            torch.nn.functional.pad(initial_latent, (initial_latent_blur,) * 4, mode='reflect'),
            kernel_size=(initial_latent_blur * 2 + 1,) * 2, stride=(1, 1))
        initial_latent = torch.nn.functional.interpolate(initial_latent, (image_height, image_width))
        initial_latent = initial_latent.to(dtype=vae.dtype, device=vae.device)
        initial_latent = vae.encode(initial_latent).latent_dist.mode() * vae.config.scaling_factor
    else:
        initial_latent = torch.zeros(size=(num_samples, 4, image_height // 8, image_width // 8), dtype=torch.float32)

    memory_management.load_models_to_gpu([unet])

    initial_latent = initial_latent.to(dtype=unet.dtype, device=unet.device)

    latents = pipeline(
        initial_latent=initial_latent,
        strength=1.0,
        num_inference_steps=int(steps),
        batch_size=num_samples,
        prompt_embeds=positive_cond,
        negative_prompt_embeds=negative_cond,
        pooled_prompt_embeds=positive_pooler,
        negative_pooled_prompt_embeds=negative_pooler,
        generator=rng,
        guidance_scale=float(cfg),
    ).images

    memory_management.load_models_to_gpu([vae])
    latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
    pixels = vae.decode(latents).sample
    B, C, H, W = pixels.shape
    pixels = pytorch2numpy(pixels)

    if highres_scale > 1.0 + eps:
        pixels = [
            resize_without_crop(
                image=p,
                target_width=int(round(W * highres_scale / 64.0) * 64),
                target_height=int(round(H * highres_scale / 64.0) * 64)
            ) for p in pixels
        ]

        pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        memory_management.load_models_to_gpu([unet])
        latents = latents.to(device=unet.device, dtype=unet.dtype)

        latents = pipeline(
            initial_latent=latents,
            strength=highres_denoise,
            num_inference_steps=highres_steps,
            batch_size=num_samples,
            prompt_embeds=positive_cond,
            negative_prompt_embeds=negative_cond,
            pooled_prompt_embeds=positive_pooler,
            negative_pooled_prompt_embeds=negative_pooler,
            generator=rng,
            guidance_scale=float(cfg),
        ).images

        memory_management.load_models_to_gpu([vae])
        latents = latents.to(dtype=vae.dtype, device=vae.device) / vae.config.scaling_factor
        pixels = vae.decode(latents).sample
        pixels = pytorch2numpy(pixels)

    for i in range(len(pixels)):
        image_path = os.path.join(save_dir, f"{unique_hex}_{i}_{relation}.png")
        image = Image.fromarray(pixels[i])
        image.save(image_path)

    return image

@torch.inference_mode()
def layout_generation(message: str, seed:int, temperature: float, top_p: float, max_new_tokens: int):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    conversation = [{"role": "system", "content": omost_canvas.system_prompt}]
    conversation.append({"role": "user", "content": message})

    memory_management.load_models_to_gpu(llm_model)
    # 只支持batch size为1， 也不用支持batch size开更大了，再大gpu都要爆了
    input_ids = llm_tokenizer.apply_chat_template(conversation,
                                                  return_tensors="pt", add_generation_prompt=True).to(llm_model.device)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature != 0 else False,
        temperature=temperature,
        top_p=top_p,
    )

    ids = llm_model.generate(**generate_kwargs)
    response = ids[0][input_ids.shape[-1]:]
    response = llm_tokenizer.decode(response, skip_special_tokens=True)
    return response

def layout_generate(caption, layout):
    total_regions = layout.split('\n\n#')
    history = [[caption,'\n\n#'.join(total_regions[:3])+'\n```']]
    unique_hex = uuid.uuid4().hex
    seed = random.randint(0, 10000)
    cprint(history[0][1], 'cyan')

    if "location='in the center'" not in history[0][1]:
        relation_list = ["ori"]
    else:
        relation_list = ["right", "left", "bottom", "up"]

    for relation in relation_list:
        tmp = history[0][1].replace("location='in the center'", f"location='in the {relation}'")
        # tmp = tmp.replace("distance_to_viewer = 2.0", "distance_to_viewer = 0")  # 这个用处似乎不大
        new_history = [[caption, tmp]]

        diffusion_fn(canvas_outputs=post_chat(new_history),
                     num_samples=1,  # todo: 生成的数量
                     seed=seed,
                     image_width=512,
                     image_height=512,
                     highres_scale=1,
                     steps=25,
                     cfg=5.0,
                     highres_steps=20,
                     highres_denoise=0.4,
                     negative_prompt='lowres, bad anatomy, bad hands, cropped, worst quality',
                     save_dir='./edit_generated_datasets/example_dataset/composition',
                     unique_hex = unique_hex,
                     relation=relation
                     )

if __name__ == "__main__":
    caption = 'generate an image of A man is cleaning the glass'
    # caption = 'A deer and a cat is playing together'
    layout = layout_generation(message=caption, seed=0, temperature=0.1, top_p=0.9, max_new_tokens=2000)
    layout_generate(caption=caption, layout=layout)