import os
import cv2
import einops
import argparse
import numpy as np
import torch
import json
import random
from diffusers import StableDiffusionPipeline
from pytorch_lightning import seed_everything
from AnyEdit_Collection.other_modules.cldm.model import create_model, load_state_dict
from AnyEdit_Collection.other_modules.cldm.ddim_hacked import DDIMSampler
from AnyEdit_Collection.other_modules.cldm.hack import disable_verbosity, enable_sliced_attention
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import *
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from tqdm import tqdm
from PIL import Image

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

def get_gpu_memory_info(device_id=0):
    torch.cuda.set_device(device_id)
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
    cached = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
    free_memory = total_memory - allocated - cached

    print(f"Device ID: {device_id}")
    print(f"Total Memory: {total_memory:.2f} MB")
    print(f"Allocated Memory: {allocated:.2f} MB")
    print(f"Cached Memory: {cached:.2f} MB")
    print(f"Free Memory: {free_memory:.2f} MB")

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    ratio = np.random.randint(11, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2])

    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512), interpolation = cv2.INTER_NEAREST).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(model, ddim_sampler, ref_image, ref_mask, tar_image, tar_mask, args):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = args.num_samples #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    guidance_scale = args.guidance_scale #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    strength = args.strength  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01) 
    ddim_steps = args.ddim_steps #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    H = args.H
    W = args.W
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": [control], 
               "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)


    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=guidance_scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']

    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image

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


def visual_inference(det_model, sam_model, model, ddim_sampler, input, instruction, output, edited_object,
                    init_image_path, ref_object, visual_input_path, edited_image_path_pre, args, test_without_model = False):
    # if use the "remove" file to transfer, then use visual_reference_add, the tar_mask is the region that removed
    total_edited_images = []
    original_size = Image.open(init_image_path).convert("RGB").size

    tar_mask_pil, tar_image_pil, tar_mask_bbox, _ = maskgeneration(det_model, sam_model, init_image_path, input, mask_mode='max', target_object=edited_object)

    if tar_mask_bbox is None:
        print('Skip: Generation target mask fails')
        return None
    # if the target is incomplete
    else:
    # Convert the bounding box mask to numpy array
        tar_bbox_mask = np.array(tar_mask_bbox)

        # Get image dimensions
        image_width, image_height = tar_image_pil.size

        # Get the bounding box coordinates
        bbox_coords = np.argwhere(tar_bbox_mask > 0)
        if len(bbox_coords) == 0:
            print('Target object is incomplete: Bounding box is empty')
            return None
        bbox_top = np.min(bbox_coords[:, 0])
        bbox_bottom = np.max(bbox_coords[:, 0])
        bbox_left = np.min(bbox_coords[:, 1])
        bbox_right = np.max(bbox_coords[:, 1])
        # Check if the bounding box touches the image edge
        if bbox_top <= 2 or bbox_bottom >= image_height - 2 or bbox_left <= 2 or bbox_right >= image_width - 2:
            print('Target object is incomplete: Bounding box touches the image edge')
            return None
        print('Target object is complete')
    
    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.png'
    tar_mask_pil.save(mask_path)
    cv2_mask_image = cv2.imread(mask_path)
    maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((10,10), np.uint8))
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate, cv2.COLOR_BGR2GRAY))

    ref_mask_pil, ref_image_pil, _, _ = maskgeneration(det_model, sam_model, visual_input_path, ref_object, mask_mode='max')
    if ref_mask_pil is None:
        print('Skip: Generation reference mask fails')
        return None
    ref_mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'_ref.png'
    ref_mask_pil.save(ref_mask_path)

    ref_cv2_mask_image = cv2.imread(ref_mask_path)
    ref_maskimage_dilate = cv2.dilate(ref_cv2_mask_image, np.ones((10,10), np.uint8))
    ref_maskimage_dilate = Image.fromarray(cv2.cvtColor(ref_maskimage_dilate, cv2.COLOR_BGR2GRAY))

    tar_image = tar_image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
    tar_mask = maskimage_dilate.resize((512, 512), resample=Image.Resampling.LANCZOS)

    ref_image = ref_image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
    ref_mask = ref_maskimage_dilate.resize((512, 512), resample=Image.Resampling.LANCZOS)

    # ref_image.save(edited_image_path_pre.replace("edited_img", "visual_img") + f'_ref.png')
    # ref_mask.save(ref_mask_path)
    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    tar_image = np.asarray(tar_image)
    tar_mask = np.asarray(tar_mask)
    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

    if not test_without_model:
        gen_image = inference_single_image(model, ddim_sampler, ref_image, ref_mask, tar_image.copy(), tar_mask, args)
        gen_image = Image.fromarray(gen_image).resize(original_size, resample=Image.Resampling.LANCZOS)
        save_edited_image_path = edited_image_path_pre + f'.png'
        gen_image.save(save_edited_image_path)
    else:
        save_edited_image_path = edited_image_path_pre + f'.png'
    
    cur_data ={"edit": instruction, "edit object": edited_object, "ref_object":ref_object, "output": output,
                        "input": input, "edit_type": "visual_reference", "reference_image_file": visual_input_path, "image_file": init_image_path, "edited_file": save_edited_image_path}
    total_edited_images.append(cur_data)
    return total_edited_images 

def parse_args():
    parser = argparse.ArgumentParser(description="Anydoor Editing Pipeline")
    parser.add_argument("--pretrained_model", type=str, required=True, help="path to checkpoint of model")
    parser.add_argument("--config", type=str, required=True, help="path to config which constructs model")
    parser.add_argument("--save_memory", type=bool, default=False, help="choose whether to save memory")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--num_samples", type=int, default=1, help="sample this often")
    parser.add_argument("--seed", type=int, default=-1, help="Seed")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="unconditional guidance scale")
    parser.add_argument("--strength", type=int, default=1, help="Control Strength")
    parser.add_argument("--ddim_steps", type=int, default=30, help="Steps")
    parser.add_argument("--groundingdino-config-file", type=str, required=True,
                        help="path to groundingdino config file")
    parser.add_argument("--groundingdino-ckpt", type=str, required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", type=str, required=True, help="path to segment anything file")
    parser.add_argument("--instruction-path", type=str, required=True, help="path to editing instruction files")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--idx", type=int, default=-1, help="specify the experiment id.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

def load_tool_model(args):
    config_file = args.groundingdino_config_file #'GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
    grounded_checkpoint = args.groundingdino_ckpt #'checkpoints/groundingdino_swinb_cogcoor.pth'  
    sam_checkpoint = args.sam_ckpt #'checkpoints/sam_vit_h_4b8939.pth'
    sd_ckpt = args.sd_ckpt # "/huggingface/stable-diffusion-v1-5"
    device = 'cuda'
    
    get_gpu_memory_info(0)
    torch.cuda.empty_cache()
    
    det_model = load_model(config_file, grounded_checkpoint, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    sd_pipe = StableDiffusionPipeline.from_pretrained(sd_ckpt, torch_dtype=torch.float16,local_files_only=True)
    
    sd_pipe = sd_pipe.to(device)

    # load model for visual reference
    model_ckpt =  args.pretrained_model
    model_config = args.config
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return sd_pipe, model, ddim_sampler, det_model, sam_model

if __name__ == '__main__': 
    test_without_model = False
    # How to construct the visual reference

    # use the instructions of "replace"
    # add the key "visual_input" with its value from the random name of jelly_cat
    # add the key "ref_object" with its value from the "animals" of the name of "visual_input" from image_category.json
    # replace the "new object" of the original replace's "edit" and "output" to the "ref_object" 
    args = parse_args()
    try:
        with open(args.json_path) as f:
            edit_instruction_data = json.load(f)
    except:
        with open(args.json_path, 'r') as f:
            edit_instruction_data = [json.loads(line) for line in f]
    if args.seed == -1:
        seed_number = random.randint(0,2147483647)
        seed_everything(seed_number)
    else:
        seed_everything(args.seed)


    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/visual_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    
    
    valid_number = 0
    success_data = []
    failure_data = []
    final_edited_results = []
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.idx}.json', 'r') as results_file:
       final_edited_results = json.load(results_file)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.idx}.json', 'r') as success_file:
        success_data = json.load(success_file)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.idx}.json', 'r') as failure_file:
        failure_data = json.load(failure_file)
    
    processed_data = len(final_edited_results) + len(failure_data)
    print('f{processed_data} has been processed')


    if not test_without_model:
        sd_model, anydoor_model, ddim_sampler, det_model, sam_model = load_tool_model(args)
    else:
        sd_model, anydoor_model, ddim_sampler, det_model, sam_model = None, None, None, None, None

    iter = 0
    for data in tqdm(edit_instruction_data):
        if iter < processed_data:
            iter = iter + 1
            print('f{iter} has been processed')
            continue
        data["edit_type"] = "visual_reference"
        input, edited_object, ref_object, visual_image_path, output, init_image_path, instruction, \
            edited_image_path_pre = return_parameters(data, args, init_image_root=args.image_path)
            # input, edited_object, ref_object, visual_image_path, output, init_image_path, instruction, edited_image_path
        ref_image_path = visual_image_path
        tar_image_path = init_image_path # come from coco image or other generated images
        
        instruction_data = None
        # ref_mask generation
        if not test_without_model:
            instruction_data = visual_inference(det_model, sam_model, anydoor_model, ddim_sampler, input, instruction, output, edited_object, init_image_path, ref_object, visual_image_path, edited_image_path_pre, args, test_without_model)

        if instruction_data is not None and len(instruction_data) > 0:
            valid_number += 1
            # final_edited_results.extend(instruction_data)
            data["edit"] = data["edit"].replace(data["ref_object"], '[V*]')
            data["output"] = data["output"].replace(data["ref_object"], '[V*]')
            success_data.append(data)
            final_edited_results.append(data)
        else:
            failure_data.append(data)
        iter = iter + 1
        print(iter) 
        if iter % 20 == 0:
            print(f"Another 20 generations done, valid editing insturction data: {valid_number}")
            with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.idx}.json', 'w') as results_file:
                json.dump(final_edited_results, results_file, indent=4)
            with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.idx}.json', 'w') as success_file:
                json.dump(success_data, success_file, indent=4)
            with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.idx}.json', 'w') as failure_file:
                json.dump(failure_data, failure_file, indent=4)  
            # break  
    
    
                
    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)    

