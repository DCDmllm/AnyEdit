import os
import random
from diffusers import AutoPipelineForText2Image, FluxPipeline
from AnyEdit_Collection.adaptive_editing_pipelines.tools.attribute_tool import StableDiffusion3InstructPix2PixPipeline
import torch
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import warnings
import json
from tqdm import tqdm
import argparse
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import is_human_variant, return_parameters, maskgeneration, generate_tags
warnings.filterwarnings("ignore")

def load_sam_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
def load_sam_image_from_Image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def image_generation_pipe(base, prompt):
    # sdxl-turbo, but not good for human
    # image = base(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    # flux
    seed = random.randint(0, 2<<32)
    image = base(
        prompt,
        output_type="pil",
        num_inference_steps=4, #use a larger number if you are using [dev]
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

def load_tool_model(args):
    sdxl_ckpt = args.sdxl_ckpt
    ultraedit_ckpt = args.sd3_ckpt
    device = 'cuda'

    det_model = load_model(args.groundingdino_config_file, args.groundingdino_ckpt, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_ckpt).to(device))
    
    # if 'FLUX' in sdxl_ckpt or 'Flux' in sdxl_ckpt or 'flux' in sdxl_ckpt: 
    #     sdxl_base = FluxPipeline.from_pretrained(sdxl_ckpt, torch_dtype=torch.bfloat16).to(device)
    # else:
    #     sdxl_base = AutoPipelineForText2Image.from_pretrained(sdxl_ckpt, torch_dtype=torch.float16, variant="fp16").to(device)
    sdxl_base = None
    edit_pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(ultraedit_ckpt, torch_dtype=torch.float16).to("cuda:1")
    return sdxl_base, edit_pipe, det_model, sam_model


def attribute_pipeline(det_model, sam_model, edit_pipe, input, edited_object, output, init_image_path,
                            edited_image_path_pre, instruction, instruction_type, kernel_size, sdxl_base=None):
    if not os.path.exists(init_image_path): 
        init_image = image_generation_pipe(sdxl_base, input)
        init_image.save(init_image_path)
    total_edited_images = []
    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
    if edited_object.split(' ')[-1] not in total_objects:
        total_objects.append(edited_object.split(' ')[-1])
    prompt = ', '.join(total_objects)
    if instruction_type in ["appearance_alter"]:
        mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                               mask_mode='merge', box_threshold=0.2, text_threshold=0.2, target_object=edited_object)
    elif instruction_type in ["color_alter"]:
        mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                mask_mode='merge', box_threshold=0.2, text_threshold=0.2, target_object=edited_object)
    if edited_object.split(' ')[-1] in ['people', 'man', 'woman', 'boy', 'girl', 'person'] or is_human_variant(edited_object.split(' ')[-1]):
        prompt = prompt + ', head, face'
        face_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                                    mask_mode='merge', box_threshold=0.2, text_threshold=0.2,
                                                    target_object=["face", "head"])
    else:
        face_mask_pil = None
    if mask_pil is None:
        print('Skip: mask is none')
        return None
    else:
        mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'
        mask_pil.save(mask_path)
        cv2_mask_image = cv2.imread(mask_path)
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones(kernel_size, np.uint8))
        mask_pil = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
        os.remove(mask_path)
    if face_mask_pil is not None:
        axy_path = edited_image_path_pre.replace("edited_img", "mask") + f'_axy.jpg'
        face_mask_pil.save(axy_path)
        face_cv2_mask_image = cv2.imread(axy_path)
        face_maskimage_dilate = cv2.dilate(face_cv2_mask_image, np.ones(kernel_size, np.uint8))
        face_maskimage_dilate = Image.fromarray(cv2.cvtColor(face_maskimage_dilate,cv2.COLOR_BGR2GRAY))
        noface_mask_pil = Image.fromarray(np.where(np.asarray(face_maskimage_dilate)==0, True, False))
        mask_pil = Image.fromarray(np.where(np.asarray(mask_pil), True, False) & np.asarray(noface_mask_pil))
        os.remove(axy_path)
    for i in range(1):
        image = edit_pipe(
            instruction,
            image=image_pil.convert("RGB"),
            mask_img=mask_pil.convert("RGB"),
            negative_prompt="bad quality, text",
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=8.0,
        ).images[0]
            
        
        edited_image = image.resize(original_size, resample=Image.Resampling.LANCZOS)
        save_edited_image_path = edited_image_path_pre + f'_{i}.jpg'
        edited_image.save(save_edited_image_path)
        
        mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'
        mask_pil.resize(original_size, resample=Image.Resampling.LANCZOS).save(mask_path)

        
        cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
        total_edited_images.append(cur_data)
    return total_edited_images 


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl model file")
    parser.add_argument("--sd3-ckpt", required=True, help="path to finetuned sd3 model file for attribute")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--idx", type=int, default=0, help="specify the experiment id.")
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

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)

    sdxl_base, edit_pipe, det_model, sam_model = load_tool_model(args)
    valid_number = 0
    success_data = []
    failure_data = []
    final_edited_results = []
    print(len(edit_instruction_data))
    for data in tqdm(edit_instruction_data):
        instruction_type = data['edit_type']
        if 'input' not in data or 'output' not in data:
            print(f'Skip: input or output in data')
            continue

        if instruction_type in ["appearance_alter", "color_alter"]:
            if 'edited object' not in data:
                print(f'Skip: no edited object in data')
                continue
            input, edited_object, output, init_image_path, instruction, edited_image_path_pre = \
                return_parameters(data, args, init_image_root=args.image_path)
            if edited_object not in input:
                print(f'Skip: no edited object in {init_image_path}')
                failure_data.append(data)
                continue

            have_exist = False
            for i in range(3):
                if os.path.exists(edited_image_path_pre+f'_{i}.jpg'):
                    save_edited_image_path = edited_image_path_pre+f'_{i}.jpg'
                    have_exist = True
                    break
            if have_exist:
                valid_number += 1
                edit_image = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                final_edited_results.append(edit_image)
                success_data.append(data)
                print(f'Skip: image existed')
                continue

            if instruction_type in ["appearance_alter"]:
                kernel_size = (30, 30)
            elif instruction_type in ["color_alter"]:
                kernel_size = (15, 15)
            else:
                raise NotImplementedError
            edit_image = attribute_pipeline(det_model=det_model, sam_model=sam_model,edit_pipe=edit_pipe,
                                            input=input, edited_object=edited_object, output=output,
                                            init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                            instruction=instruction, instruction_type=instruction_type, kernel_size=kernel_size, sdxl_base=sdxl_base)
        else:
            raise NotImplementedError

        if edit_image is not None:
            valid_number += 1
            final_edited_results.extend(edit_image)
            success_data.append(data)
        else:
            failure_data.append(data)
    # logging.info(f"valid editing insturction data: {valid_number}")
    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)