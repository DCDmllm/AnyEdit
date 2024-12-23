import os
import torch
import json
from tqdm import tqdm
from diffusers import DDIMScheduler, StableDiffusionPipeline
from AnyEdit_Collection.other_modules.masactrl.masactrl_utils import regiter_attention_editor_diffusers
from AnyEdit_Collection.other_modules.masactrl.masactrl import MutualSelfAttentionControl
from torchvision.utils import save_image
# from masactrl.masactrl_processor import register_attention_processor
import spacy
import argparse
from pytorch_lightning import seed_everything
import random

def consistent_synthesis(model, prompts, init_image_path, edit_image_path, img_size=512, model_type="SD"):
    seed = random.randint(0, 2 <<16)
    seed_everything(seed)
    # initialize the noise map
    if model_type == "SD":
        start_code = torch.randn([1, 4, 64, 64], device=device, dtype=torch.float16)
        STEP_LIST = [5]
        LAYER_LIST = [12]
    elif model_type == "SDXL":
        start_code = torch.randn([1, 4, 128, 128], device=device, dtype=torch.float16)
        STEP_LIST = [5]
        LAYER_LIST = [64]
    else:
        raise NotImplementedError
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    for STEP in STEP_LIST:
        for LAYER in LAYER_LIST:
            # hijack the attention module
            guidance_scale = random.choice([6.5,7.0,7.5,8.0,8.5])
            editor = MutualSelfAttentionControl(STEP, LAYER, model_type=model_type)
            regiter_attention_editor_diffusers(model, editor)
            # inference the synthesized image
            if model_type == "SD":
                image_masactrl = model(prompts, latents=start_code, guidance_scale=guidance_scale).images
                image_masactrl[0].save(init_image_path)
                image_masactrl[1].save(edit_image_path)
            else:
                image_masactrl = model(prompts, latents=start_code, guidance_scale=guidance_scale).images
                image_masactrl[0].save(init_image_path)
                image_masactrl[1].save(edit_image_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--model-path", required=True, help="path to sdxl model file")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", type=str, default=None, help="the path of instructions.")
    parser.add_argument("--start-idx", type=str, default='-1', help="specify the experiment id.")
    parser.add_argument("--end-idx", type=str, default='-1', help="specify the experiment id.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    try:
        with open(args.json_path) as f:
            edit_instruction_data = json.load(f)
    except:
        with open(args.json_path, 'r') as f:
            edit_instruction_data = [json.loads(line) for line in f]
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)

    if args.start_idx != '-1' and args.end_idx != '-1':
        print(args.start_idx, args.end_idx)
        st_idx = int(args.start_idx)
        ed_idx = int(args.end_idx)
        edit_instruction_data = edit_instruction_data[st_idx:ed_idx]
    print(len(edit_instruction_data))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    weight_dtype = torch.float16
    model = StableDiffusionPipeline.from_single_file(args.model_path, scheduler=scheduler, variant="fp16", torch_dtype=weight_dtype, local_files_only=True).to(device)

    total_data = []
    valid_number = 0
    for i, data in enumerate(tqdm(edit_instruction_data[:2])):
        # image_file = str(i)
        if 'image_file' in data:
            image_file = data['image_file']
        else:
            image_file = f"combination_action_{i}.jpg"
        edit_type = data['edit_type']

        # init_image_path = f'{args.instruction_path}/{args.instruction_type}/input_img/init_{edit_type}_{image_file}.png'
        # edit_image_path = f'{args.instruction_path}/{args.instruction_type}/edited_img/edited_{edit_type}_{image_file}.png'
        init_image_path = f'{args.instruction_path}/{args.instruction_type}/input_img/{image_file}'
        edit_image_path = f'{args.instruction_path}/{args.instruction_type}/edited_img/{image_file}'

        # if os.path.exists(init_image_path):
        #     continue

        is_valid = True
        # target_actions = generate_tags(data['edit'])
        # origin_actions = generate_tags(data['input'])
        # for action in target_actions['verb']:
        #     if action in origin_actions['verb']:
        #         is_valid = False
        #         break
        if is_valid:
            prompt1 = data['input'][:-1]+', realistic photo, best quality' if data['input'][-1] == '.' else  data['input']+', realistic photo, best quality'
            prompt2 = data['output'][:-1]+', realistic photo, best quality' if data['output'][-1] == '.' else  data['output']+', realistic photo, best quality'
            print(prompt1)
            print(prompt2)
            prompts = [prompt1, prompt2]
            data['visual_input'] = None
            data["image_file"] = init_image_path
            data["edited_file"] = edit_image_path
            consistent_synthesis(model, prompts, init_image_path, edit_image_path, model_type="SD")
            total_data.append(data)
            valid_number += 1
    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.start_idx}_{args.end_idx}.json', 'w') as results_file:
        json.dump(total_data, results_file, indent=4)