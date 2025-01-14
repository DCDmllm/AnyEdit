import os
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
import torch
from AnyEdit_Collection.other_modules.ip_adapter import IPAdapterXL
from AnyEdit_Collection.other_modules.ip_adapter.utils import register_cross_attention_hook
from PIL import Image, ImageChops
from PIL import ImageEnhance
import numpy as np
import json
import argparse
from tqdm import tqdm
import random
# GroundingDINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
# segment anything
from segment_anything import build_sam, SamPredictor
# DPT
from AnyEdit_Collection.other_modules.DPT.dpt.models import DPTDepthModel
from visual_condition_tool import run_depth
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import generate_tags, maskgeneration
import warnings
warnings.filterwarnings("ignore")

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def load_tool_model(args):
    device = "cuda"
    image_encoder_path = args.image_encoder_path
    ip_ckpt = args.ip_ckpt
    controlnet_path = args.controlnet_path
    dpt_model_path = args.dpt_model_path

    torch.cuda.empty_cache()

    det_model = load_model(args.groundingdino_config_file, args.groundingdino_ckpt, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=args.sam_ckpt).to(device))
    # load SDXL pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        args.sdxl_ckpt,
        controlnet=controlnet,
        use_safetensors=True,
        torch_dtype=torch.float16,
        add_watermarker=False,
        variant='fp16'
    ).to(device)
    pipe.unet = register_cross_attention_hook(pipe.unet)

    ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

    dpt_model = DPTDepthModel(
        path=dpt_model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    dpt_model = dpt_model.to(device)

    return det_model, sam_model, ip_model, dpt_model


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl/sd1.5 model file")
    parser.add_argument("--image-encoder-path", required=True, help="path to corresponding image encoder") 
    parser.add_argument("--ip-ckpt", required=True, help="path to ip-adapter for editing")
    parser.add_argument("--controlnet-path", required=True, help="path to controlnet for depth") 
    parser.add_argument("--dpt-model-path", required=True, help="path to dpt model for editing")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--options", nargs="+", help="in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    edit_instruction_data = []
    try:
        with open(args.json_path) as f:
            for line in f:
                edit_instruction_data.append(json.loads(line.strip()))
    except:
        edit_instruction_data = json.load(open(args.json_path))

    det_model, sam_model, ip_model, dpt_model = load_tool_model(args)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/depth', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/visual_input', exist_ok=True)
    
    valid_data = []
    iter = 0
    for data in tqdm(edit_instruction_data[:2]):
        iter = iter + 1
        edited_object = data['edited object']
        image_path_pre = data['image_file'].split('/')[-1].split('.')[0]

        target_image_path = os.path.join(args.image_path, data['image_file'])
        
        # need to put all material image into the path 'xxxx/xxxx/materials'
        material_exemplar_root = f"{args.instruction_path}/{args.instruction_type}/materials"
        material_name = random.choice(os.listdir(material_exemplar_root))
        
        material_exemplar_path = f"{material_exemplar_root}/{material_name}"
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        all_files = os.listdir(material_exemplar_path)
        image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_extensions]

        if image_files:
            selected_image = random.choice(image_files)
            material_image_path = os.path.join(material_exemplar_path, selected_image)
            ip_image = Image.open(material_image_path).convert("RGB")
            output_visual_path = f"{args.instruction_path}/{args.instruction_type}/visual_input/{material_name}_{image_path_pre}.jpg"
            ip_image.save(output_visual_path)
            data["visual_input"] = output_visual_path
            print(f"Selected material image path: {material_image_path}")
        else:
            print("No image files found in the selected folder.")
            continue
        output_depth_image = f"{args.instruction_path}/{args.instruction_type}/depth/{image_path_pre}"
        output_mask_image = f"{args.instruction_path}/{args.instruction_type}/mask/{image_path_pre}.jpg"
        output_image_path = f"{args.instruction_path}/{args.instruction_type}/edited_img/{material_name}_{image_path_pre}.jpg"
        print(f'Editing: {target_image_path}')
        
        total_objects = generate_tags(data['input'])['nouns']
        if edited_object not in total_objects:
            print(f'Skip: input or output in data')
            continue
        det_prompt = ', '.join(total_objects)

        target_mask, target_image, _, _ = maskgeneration(det_model, sam_model, target_image_path, det_prompt, target_object=edited_object)
        if target_mask is not None:
            target_mask = target_mask.convert('L').convert('RGB')
            target_mask.save(output_mask_image)
        else:
            print(f"Failed to generate mask for {target_image_path}")
            continue

        # Generate random noise for the size of the image
        noise = np.random.randint(0, 256, target_image.size + (3,), dtype=np.uint8)
        noise_image = Image.fromarray(noise)
        mask_target_img = ImageChops.lighter(target_image, target_mask)
        invert_target_mask = ImageChops.invert(target_mask)


        gray_target_image = target_image.convert('L').convert('RGB')
        gray_target_image = ImageEnhance.Brightness(gray_target_image)

        # Adjust brightness
        # The factor 1.0 means original brightness, greater than 1.0 makes the image brighter. Adjust this if the image is too dim
        factor = 1.0  # Try adjusting this to get the desired brightness

        gray_target_image = gray_target_image.enhance(factor)
        grayscale_img = ImageChops.darker(gray_target_image, target_mask)
        img_black_mask = ImageChops.darker(target_image, invert_target_mask)
        grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img)
        init_img = grayscale_init_img
        
        run_depth(target_image_path, output_depth_image, dpt_model)  # save depth image for other visual reference
        np_image = np.array(Image.open(output_depth_image+".png"))

        np_image = (np_image / 256).astype('uint8')

        depth_map = Image.fromarray(np_image).resize((1024,1024))

        init_img = init_img.resize((1024,1024))
        mask = target_mask.resize((1024, 1024))
        
        num_samples = 1
        images = ip_model.generate(pil_image=ip_image, image=init_img, control_image=depth_map, mask_image=mask, controlnet_conditioning_scale=0.9, num_samples=num_samples, num_inference_steps=30, seed=42)
        
        
        images[0].save(output_image_path)

        # data["visual_input"] = material_exemplar_path
        data['image_file'] = target_image_path
        data["edited_path"] = output_image_path
        data["material"] = material_name
        valid_data.append(data)

    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_-1_-1.json', 'w') as f:
        json.dump(valid_data, f, indent=4)
