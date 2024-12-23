import json
import os
import torch
import random
from diffusers import FluxPipeline
from diffusers.utils import load_image
from tqdm import tqdm
from pytorch_lightning import seed_everything

if __name__ == "__main__":
    device = 'cuda'

    pipe = FluxPipeline.from_pretrained("./checkpoints/foundation_models/flux", torch_dtype=torch.bfloat16, local_files_only=True).to(device)
    # pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

    with open('./AnyEdit/instructions/textual_preflitered.jsonl.jsonl', 'r') as f:
        edit_instruction_data = [json.loads(line) for line in f]
    os.makedirs('anyedit_datasets/textual_change/input_img', exist_ok=True)
    os.makedirs('anyedit_datasets/textual_change/edited_img', exist_ok=True)
    valid_instructions = []
    for instruction in tqdm(edit_instruction_data):
        prompt0 = instruction['input']
        prompt1 = instruction['output']
        image0_path = os.path.join('anyedit_datasets/textual_change/', instruction['image_file'])
        image1_path = os.path.join('anyedit_datasets/textual_change/', instruction['edited_file'])
        if os.path.exists(image0_path) and os.path.exists(image1_path):
            instruction['image_file'] = os.path.join('anyedit_datasets/textual_change', instruction['image_file'])
            instruction['edited_file'] = os.path.join('anyedit_datasets/textual_change', instruction['edited_file'])
            valid_instructions.append(instruction)
            continue
        seed = random.randint(0, 2<<16)
        seed_everything(seed)
        guidance_scale = random.choice([1.0,1.5,2.0,2.5,3.0,3.5,4.0])
        image0 = pipe(
            prompt0,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        image0.save(image0_path)
        image1 = pipe(
            prompt1,
            guidance_scale=guidance_scale,
            height=512,
            width=512,
            num_inference_steps=4,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        image1.save(image1_path)
        instruction['image_file'] = os.path.join('./AnyEdit/data/textual_change', instruction['image_file'])
        instruction['edited_file'] = os.path.join('./AnyEdit/data/textual_change', instruction['edited_file'])
        valid_instructions.append(instruction)
    print(len(valid_instructions))
    with open(f'./AnyEdit/data/textual_change/final_edit_results_-1_-1.json', 'w') as results_file:
        json.dump(valid_instructions, results_file, indent=4)