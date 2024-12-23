import os
import random
from diffusers import AutoPipelineForText2Image, FluxPipeline
from AnyEdit_Collection.adaptive_editing_pipelines.tools.global_tool import LocalEditor
import torch, torchvision
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
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import return_parameters, maskgeneration, generate_tags
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

def cropimage2image(original_image, mask, background_image, scale):
    original_image_array = np.array(original_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]
    extracted_content_mask=mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[min_y:min_y+scaled_h, min_x:min_x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def load_tool_model(args):
    sdxl_ckpt = args.sdxl_ckpt
    ip2p_ckpt = args.ip2p_ckpt
    device = 'cuda'

    det_model = load_model(args.groundingdino_config_file, args.groundingdino_ckpt, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_ckpt).to(device))
    # if 'FLUX' in sdxl_ckpt or 'Flux' in sdxl_ckpt or 'flux' in sdxl_ckpt: 
    #     sdxl_base = FluxPipeline.from_pretrained(sdxl_ckpt, torch_dtype=torch.bfloat16).to(device)
    # else:
    #     sdxl_base = AutoPipelineForText2Image.from_pretrained(sdxl_ckpt, torch_dtype=torch.float16, variant="fp16").to(device)
    sdxl_base = None
    instructpix2pix_pipe = LocalEditor(ip2p_ckpt, ip2p_ckpt) # if mask_threshold==0.0, is naive instructPix2Pix

    return sdxl_base, instructpix2pix_pipe, det_model, sam_model

def clip_filtering_image(image_elements, text, clip_processor, clip_model, device="cuda"):
    clip_inputs = clip_processor(text=[text], images=image_elements, return_tensors="pt").to(device)
    clip_outputs = clip_model(**clip_inputs)
    image_features = clip_outputs['image_embeds'] / clip_outputs['image_embeds'].norm(dim=-1, keepdim=True)
    text_features = clip_outputs['text_embeds'] / clip_outputs['text_embeds'].norm(dim=-1, keepdim=True)
    clip_similarity = 100. * image_features @ text_features.T
    results = torch.max(clip_similarity, dim=0)
    return results[0], results[1].item()

def convert_to_three_channels(image_tensor):
    image_tensor_three_channels = torch.cat([image_tensor] * 3, dim=0)
    return image_tensor_three_channels

def mask_crop_ip2p_pipeline(det_model, sam_model, instructpix2pix_pipe, input, edited_object, output, init_image_path,
                            edited_image_path_pre, instruction, instruction_type, sdxl_base=None):
    if not os.path.exists(init_image_path): 
        init_image = image_generation_pipe(sdxl_base, input)
        init_image.save(init_image_path)
    total_edited_images = []
    original_size = Image.open(init_image_path).convert("RGB").size
    if instruction_type == 'tone_transfer':
        for i in range(1):
            image_tensor0 = torchvision.io.read_image(init_image_path).float() / 255 * 2.0 - 1
            image_tensor  = torchvision.transforms.Resize([512, 512])(image_tensor0)
            if image_tensor.shape[0] == 1:  # singel channel
                image_tensor = convert_to_three_channels(image_tensor)
            edited_image = instructpix2pix_pipe(image_tensor, 
                                                    edit = instruction, 
                                                    check_size = False,
                                                    scale_txt = 8.0,
                                                    scale_img = 0.9,
                                                    mask_threshold = 0.0,  # naive ip2p
                                                    return_heatmap = False)
            edited_image = torchvision.transforms.Resize([original_size[1], original_size[0]])(edited_image)
            save_edited_image_path = edited_image_path_pre + f'.jpg'
            torchvision.utils.save_image(tensor = edited_image, 
                                    normalize = True,
                                    scale_each = True,
                                    fp = save_edited_image_path)
            cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
            total_edited_images.append(cur_data)
        return total_edited_images 

    total_objects = generate_tags(input)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
    if edited_object.split(' ')[-1] not in total_objects:
        total_objects.append(edited_object.split(' ')[-1])
    prompt = ', '.join(total_objects)
    mask_pil, image_pil, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt,
                                               mask_mode='merge', box_threshold=0.2, text_threshold=0.2, target_object=edited_object)

    if mask_pil is None:
        return None
    
    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'
    mask_pil.save(mask_path)
    cv2_mask_image = cv2.imread(mask_path)  

    if instruction_type == 'color_alter':
        maskimage_dilate = cv2.GaussianBlur(cv2_mask_image, (5, 5), 0)  
    else:
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((10, 10), np.uint8))
        maskimage_dilate = cv2.GaussianBlur(cv2_mask_image, (5,5), 0)   
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    image_tensor0 = torchvision.io.read_image(init_image_path).float() / 255 * 2.0 - 1
    image_tensor = torchvision.transforms.Resize([512, 512])(image_tensor0)

    if image_tensor.shape[0] == 1: # singel channel
        image_tensor = convert_to_three_channels(image_tensor)

    for i in range(1):
        edited_image = instructpix2pix_pipe(image_tensor, 
                                            edit = instruction,
                                            check_size = False,
                                            scale_txt = 8.0,
                                            scale_img = 0.9,
                                            mask_threshold = 0.0,  # naive ip2p
                                            return_heatmap = False)
        torchvision.utils.save_image(tensor = edited_image, 
                                normalize = True,
                                scale_each = True,
                                fp = edited_image_path_pre + f'_ip2p.jpg')
        edited_image = Image.open(edited_image_path_pre + f'_ip2p.jpg').convert('RGB')
        edited_image = cropimage2image(edited_image, maskimage_dilate, image_pil, scale=1) 
            
        os.remove(edited_image_path_pre + f'_ip2p.jpg')
        edited_image = edited_image.resize(original_size, resample=Image.Resampling.LANCZOS)
        save_edited_image_path = edited_image_path_pre + f'.jpg'
        edited_image.save(save_edited_image_path)
        cur_data = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
        total_edited_images.append(cur_data)
    return total_edited_images 

def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl model file")
    parser.add_argument("--ip2p-ckpt", required=True, help="path to instructpix2pix model file")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
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

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)

    sdxl_base, instructpix2pix_pipe, det_model, sam_model = load_tool_model(args)
    valid_number = 0
    success_data = []
    failure_data = []
    final_edited_results = []
    if args.start_idx != '-1' and args.end_idx != '-1':
        print(args.start_idx, args.end_idx)
        st_idx = int(args.start_idx)
        ed_idx = int(args.end_idx)
        edit_instruction_data = edit_instruction_data[st_idx:ed_idx]
    print(len(edit_instruction_data))
    for data in tqdm(edit_instruction_data):
        instruction_type = data['edit_type']
        if 'input' not in data or 'output' not in data:
            print(f'Skip: input or output in data')
            continue

        if instruction_type == "color_alter":
            if 'edited object' not in data:
                print(f'Skip: no edited object in data')
                continue
            input, edited_object, output, init_image_path, instruction, edited_image_path_pre = \
                return_parameters(data, args, init_image_root=args.image_path)
            if edited_object.split(' ')[-1] not in input.strip('.').replace(',','').split(' '):
                print(f'Skip: no obvious edited object in {init_image_path}')
                failure_data.append(data)
                continue
            # continue generation for NVCC
            have_exist = False
            # if os.path.exists(edited_image_path_pre + f'.jpg'):
            #     save_edited_image_path = edited_image_path_pre + f'.jpg'
            #     have_exist = True
            if have_exist:
                print(f'Skip: output image exist')
                valid_number += 1
                edit_image = {"edit": instruction, "edit object": edited_object, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                final_edited_results.append(edit_image)
                success_data.append(data)
                continue

            edit_image = mask_crop_ip2p_pipeline(det_model=det_model, sam_model=sam_model,    instructpix2pix_pipe=instructpix2pix_pipe, input=input, edited_object=edited_object, output=output,
                                                init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,
                                                instruction=instruction, instruction_type=instruction_type, sdxl_base=sdxl_base)

        elif instruction_type == "tone_transfer":
            input, output, init_image_path, instruction, edited_image_path_pre = \
                return_parameters(data, args, init_image_root=args.image_path)
            # continue generation for NVCC
            have_exist = False
            if os.path.exists(edited_image_path_pre + f'.jpg'):
                save_edited_image_path = edited_image_path_pre + f'.jpg'
                have_exist = True
            else:
                for i in range(3):
                    if os.path.exists(edited_image_path_pre+f'_{i}.jpg'):
                        save_edited_image_path = edited_image_path_pre+f'_{i}.jpg'
                        have_exist = True
                        break
            if have_exist:
                valid_number += 1
                edit_image = {"edit": instruction, "edit object": None, "output": output, "input": input, "edit_type": instruction_type, "image_file": init_image_path, "edited_file": save_edited_image_path}
                final_edited_results.append(edit_image)
                success_data.append(data)
                continue
            edited_object = ' '
            edit_image = mask_crop_ip2p_pipeline(det_model=det_model, sam_model=sam_model, instructpix2pix_pipe=instructpix2pix_pipe, input=input, edited_object=edited_object, output=output,
            init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre, instruction=instruction, instruction_type=instruction_type, sdxl_base=sdxl_base)
        else:
            raise NotImplementedError

        if edit_image is not None:
            valid_number += 1
            final_edited_results.extend(edit_image)
            success_data.append(data)
        else:
            failure_data.append(data)

    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.start_idx}_{args.end_idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.start_idx}_{args.end_idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.start_idx}_{args.end_idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)