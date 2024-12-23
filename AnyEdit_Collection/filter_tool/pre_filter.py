import os

import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import clip
import json
from PIL import Image
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import random
import sys
sys.path.insert(0, './AnyEdit_Collection')
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import generate_tags, is_human_variant, union_generation, load_model   
from transformers import AutoProcessor, Blip2ForConditionalGeneration

def clip_score(image, text, clip_model, clip_preprocess, device):
    images = [image]
    texts = [text]
    try:
        image_inputs = torch.stack([clip_preprocess(image) for image in images]).to(device)
        text_input = clip.tokenize(texts).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_inputs)
            text_features = clip_model.encode_text(text_input)

        clip_score = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
        return clip_score[0][0]
    except:
        return -1

def load_tool_model(device, eval_mode):
    #  For CLIP score and Aesthetic score
    clip_model, clip_preprocess = clip.load('ViT-L/14', device=device)
    class MLP(pl.LightningModule):
        def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
            super().__init__()
            self.input_size = input_size
            self.xcol = xcol
            self.ycol = ycol
            self.layers = nn.Sequential(
                nn.Linear(self.input_size, 1024),
                #nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 128),
                #nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                #nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(64, 16),
                #nn.ReLU(),

                nn.Linear(16, 1)
            )

        def forward(self, x):
            return self.layers(x)

        def training_step(self, batch, batch_idx):
                x = batch[self.xcol]
                y = batch[self.ycol].reshape(-1, 1)
                x_hat = self.layers(x)
                loss = F.mse_loss(x_hat, y)
                return loss
        
        def validation_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            return optimizer
    aesthetic_model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

    aesthetic_model_paras = torch.load("./checkpoints/flitering_model/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
    aesthetic_model.load_state_dict(aesthetic_model_paras)

    aesthetic_model.to(device)
    aesthetic_model.eval()  
    clip_model = None
    clip_preprocess = None
    aesthetic_model = None
    assert eval_mode in ['blip2', 'vila']
    if eval_mode=='blip2':
        ## blip2 evaluation
        blip2_vis_processors = AutoProcessor.from_pretrained("./checkpoints/flitering_model/blip-t5", local_files_only=True)
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("./checkpoints/flitering_model/blip-t5", torch_dtype=torch.float16, local_files_only=True).to(device)
        return clip_model, clip_preprocess, aesthetic_model, blip2_model, blip2_vis_processors, None
    else:
        ## vila evaluation
        from AnyEdit_Collection.other_modules.vila.model.builder import load_pretrained_model
        from AnyEdit_Collection.other_modules.vila.utils import disable_torch_init
        from AnyEdit_Collection.other_modules.vila.mm_utils import get_model_name_from_path
        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model("./checkpoints/flitering_model/vila-1.5-3b",
                                               get_model_name_from_path("./checkpoints/flitering_model/vila-1.5-3b"), None)
        model.eval()
        return clip_model, clip_preprocess, aesthetic_model, model, image_processor, tokenizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def pre_filter(instructions, instruction_type, device, image_root, clip_model, clip_preprocess):
    print(instruction_type, len(instructions))
    object_size_count = 0 
    aesthetic_count = 0
    clip_count = 0
    instruction_count = 0
    total = 0
    valid_instructions = []
    config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    grounded_checkpoint = "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth"
    det_model = load_model(config_file, grounded_checkpoint, device=device)
    for instruction in tqdm(instructions):
        image_file = instruction["image_file"]
        number_part = image_file.split('_')[-1]
        
        image_path = f'{image_root}/COCO_train2014_{number_part}'
        if not os.path.exists(image_path):
            continue
        total += 1
        pil_image = Image.open(image_path)   

        # instruction_level filter
        if instruction_type == 'background_change':
            if 'new background' not in instruction:
                continue
        elif instruction_type == 'appearance_alter':
            if 'edited object' not in instruction or instruction['edited object'].split(' ')[-1] not in instruction['input'].replace(',', '').strip('.').split(' ') or 'color' in instruction['edit'].strip().split(' '):
                continue 
        elif instruction_type == 'remove' or instruction_type == 'replace':           
            # if 'edited object' not in instruction or instruction['edited object'] not in instruction['input'].strip().split(' '):
            if 'edited object' not in instruction:
                continue

        width, height = pil_image.size
        if max(width, height) / min(width, height) <= 2: # 长宽
            if clip_score(image=pil_image, text=instruction["input"],
                          clip_model=clip_model, clip_preprocess=clip_preprocess, device=device)>0.1: # clip 判断
                clip_count += 1
                # # 美学分数判断
                image = clip_preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                im_emb_arr = normalized(image_features.cpu().detach().numpy() )
                prediction = aesthetic_model(torch.from_numpy(im_emb_arr).to(device).type(torch.float32))   
                prediction_list = prediction.tolist()
                grade = prediction[0][0].item()
                if grade > 2: # 美学分数
                    aesthetic_count += 1
                    if 'edited object' in instruction and instruction_type in ['remove', 'replace', 'appearance_alter', 'color_alter']:
                        object_ratio = union_generation(det_model, image_path, instruction["edited object"], device=device)
                        if object_ratio < 0.95: # 物体大小
                            object_size_count += 1
                            valid_instructions.append(instruction)
                    else: # background_change or tone_transfer or add
                        object_size_count += 1
                        valid_instructions.append(instruction)
                            
        if max(width, height) / min(width, height) <= 2: # 长宽
                    if 'edited object' in instruction and instruction_type in ['remove', 'replace', 'appearance_alter', 'color_alter']:
                        # object_ratio = union_generation(det_model, image_path, instruction["edited object"], device=device)
                        object_ratio = 0.5
                        if object_ratio < 0.95: # 物体大小
                            object_size_count += 1
                            valid_instructions.append(instruction)
                    else: # background_change or tone_transfer or add
                        object_size_count += 1
                        valid_instructions.append(instruction)
    print("Total", total)
    print("Selected by instruction", instruction_count)
    print("Selected by clip score", clip_count)
    print("Selected by aesthetic score", aesthetic_count)
    print("Selected by object size", object_size_count)
    
    return valid_instructions

def color_prefilter(instructions):
    # balanced_color_alter
    basic_colors = ["red", "blue", "green", "yellow", "black", "purple", "white", "pink", "orange", "gray", "brown", "gold", "silver"]
    rare_colors = [
        "Violet", "Indigo", "Magenta", "Cyan", "Teal", "Lime", "Olive",
        "Navy", "Maroon", "Coral", "Turquoise", "Tan", "Salmon", "Beige", "Lavender", "Mint", "Aqua"
    ]
    total_color = [basic_colors, rare_colors]
    color_probability = [0.8, 0.2]
    valid_instructions = []
    for instruction in instructions:
        color_list = random.choices(total_color, color_probability, k=1)[0]
        color = random.choice(color_list)
        try:
            cur_color = instruction['edit'].split('to ')[1]
        except:
            continue
        cur_color = instruction['edit'].split('to ')[1]
        instruction['edit'] = instruction['edit'].split('to ')[0] + f'to {color.lower()}'
        instruction['output'] = instruction['output'].replace(cur_color, color.lower())
        valid_instructions.append(instruction)
    return valid_instructions

def appearance_prefilter(edit_instructions):
    invalid_words = ['people', 'man', 'woman', 'boy', 'girl', 'person']
    invalid_opearate = ['size', 'bigger', 'smaller', 'longer', 'shorter']
    valid_instructions = []
    for instruction in edit_instructions:
        invalid_instruction = False
        for word in invalid_words:
            if word in instruction['edit'].split('to ')[-1] or word in instruction['edit'].split('into ')[-1]:
                invalid_instruction = True
                break
        for word in invalid_opearate:
            if word in instruction['edit']:
                invalid_instruction = True
                break
        if not invalid_instruction:
            valid_instructions.append(instruction)
    return valid_instructions

def replace_prefilter(edit_instructions):
    invalid_words = ['people', 'man', 'woman', 'boy', 'girl', 'person', 'adult', 'child']
    valid_instructions = []
    lemmatizer = WordNetLemmatizer()
    for instruction in edit_instructions:
        edited_object = instruction['edited object']
        new_object = instruction['new object']
        original_new_object = lemmatizer.lemmatize(new_object.split(' ')[-1], pos='n')
        if new_object.split(' ')[-1] in invalid_words or is_human_variant(new_object.split(' ')[-1]) or original_new_object in invalid_words or is_human_variant(original_new_object):
            continue
        else:
            valid_instructions.append(instruction)
    return valid_instructions

def action_prefilter(edit_instructions):
    invalid_words = ['people', 'man', 'woman', 'boy', 'girl', 'person', 'adult', 'child']
    valid_instructions = []
    lemmatizer = WordNetLemmatizer()
    for instruction in tqdm(edit_instructions):
        target_action = instruction['edit'].split('to ')[-1].split(' ')[0]
        if target_action in ['be', 'being', 'have', 'having']:
            invalid_instruction = True
        else:
            invalid_instruction = False
            action = instruction['edit'].split('to ')[-1].split(' ')[0]
            original_action = lemmatizer.lemmatize(action, pos='v')
            update_input = []
            update_output = []
            for input_sentence, sentence in zip(instruction['input'].split(', '), instruction['output'].split(', ')):
                cut = False
                for w in sentence.strip('.').split(' '):
                    if lemmatizer.lemmatize(w, pos='v') == original_action:
                        input_actions = generate_tags(input_sentence)["verb"]
                        if len(input_actions) == 0 or original_action in input_actions or action in input_actions:
                            invalid_instruction = True
                        cut = True
                        break
                update_input.append(input_sentence)
                update_output.append(sentence)
                if cut:
                    break
            update_input = ', '.join(update_input)
            update_output = ', '.join(update_output)
        if not invalid_instruction:
            instruction['input'] = update_input
            instruction['output'] = update_output
            valid_instructions.append(instruction)
    return valid_instructions

def blip2_evaluation_answers_generation(image_root, image_infos, model, processors, device, recaption=False):
    blip2_candidate_answers = []
    for info in tqdm(image_infos):
        image_path = os.path.join(image_root, info["image_file"])
        image_pil = Image.open(image_path).convert("RGB")
        question = "Question: What's the background of this image? Answer: "
        inputs = processors(image_pil, question, return_tensors="pt").to(device, torch.float16)
        out = model.generate(**inputs)
        blip2_candidate_answer = processors.decode(out[0], skip_special_tokens=True)

        if recaption:
            caption_question = "Describe the image in detail."
            caption_inputs = processors(image_pil, caption_question, return_tensors="pt").to(device, torch.float16)
            caption_out = model.generate(**caption_inputs)
            caption = processors.decode(caption_out[0], skip_special_tokens=True)
            caption = caption.replace(',','.')
            caption = caption.split('.')[0]+'.'
        else:
            caption = info['input']
        blip2_candidate_answers.append([blip2_candidate_answer, caption])

    return blip2_candidate_answers


from AnyEdit_Collection.other_modules.vila.constants import IMAGE_TOKEN_INDEX
from AnyEdit_Collection.other_modules.vila.conversation import SeparatorStyle, conv_templates
from AnyEdit_Collection.other_modules.vila.mm_utils import (KeywordsStoppingCriteria, process_images, tokenizer_image_token)
from tqdm import tqdm
def vila_evaluation_answers_generation(image_root, image_infos, model, processors, device, tokenizer):
    vila_candidate_answers = []
    for info in tqdm(image_infos):
        image_path = os.path.join(image_root, info["image_file"])
        images_pil = [Image.open(image_path).convert("RGB")]

        conv = conv_templates['vicuna_v1'].copy()
        conv.append_message(conv.roles[0], "<image>\nPlease directly answer in brief words, the background of this picture is: ")
        conv.append_message(conv.roles[1], None)
        input = conv.get_prompt()

        images_tensor = process_images(images_pil, processors, model.config).to(model.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(input, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(
            0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=False,  # t>0, need True
                temperature=0.1,
                top_p=None,
                num_beams=1,
                max_new_tokens=30,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        vila_candidate_answers.append([outputs, info['input']])

    return vila_candidate_answers

def background_prefilter(image_root, instructions, model, processors, device, tokenizer):
    if tokenizer is not None: # vila
        candidate_answers = vila_evaluation_answers_generation(image_root, instructions, model, processors, device, tokenizer)
    else: # blip2
        candidate_answers = blip2_evaluation_answers_generation(image_root, instructions, model, processors, device)

    prefilter_edit_instructions = []
    for image_info, blip2_check in zip(instructions, candidate_answers):
        # phrase1: if 'background' in exist caption (is evident in the image) and 'background' not in the edit instruction (not same as new background) --> valid edit instruction
        cur_background = blip2_check[0]
        new_background = image_info['new background']
        if new_background.split(' ')[-1] in cur_background.strip('.').split(' ') or new_background.split(' ')[-1] in image_info['input'].strip('.').split(' '):
            continue
        else:
            valid_foreground = []
            total_objects = generate_tags(blip2_check[1])['nouns']
            for obj in total_objects:
                if obj not in cur_background.strip('.').split(' ') and obj not in ['kitchen', 'room', 'bathroom', 'bedroom', 'wall']:
                    valid_foreground.append(obj)  
            if len(valid_foreground) > 0:
                image_info['foreground'] = valid_foreground
                image_info['original_background'] = cur_background
                prefilter_edit_instructions.append(image_info)
    return prefilter_edit_instructions


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--instruction-path", help="path to instruction path",
                        default='')
    parser.add_argument("--instruction-type", help="specify the instruction type.",
                        default='action_change')
    parser.add_argument("--image-root", default='')
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
    device = "cuda"
    clip_model, clip_preprocess, aesthetic_model, blip2_model, blip2_vis_processors, tokenizer = load_tool_model(device, eval_mode='vila')
    try:
        edit_instructions = json.load(open(args.instruction_path))
        line_mode = True
    except:
        with open(args.instruction_path, 'r') as f:
            edit_instructions = [json.loads(line) for line in f]
        line_mode = False

    print("original instruction: ", len(edit_instructions))
    # pre_filter
    if args.instruction_type == 'color_alter':
        filter_edit_instructions = color_prefilter(edit_instructions)
    elif args.instruction_type == 'background_change':
        filter_edit_instructions = background_prefilter(image_root=args.image_root,
                                                        instructions=edit_instructions, model=blip2_model,
                                                        processors=blip2_vis_processors, device=device, tokenizer=tokenizer)
    elif args.instruction_type == 'appearance_alter':
        filter_edit_instructions = appearance_prefilter(edit_instructions)
    elif args.instruction_type == 'replace':
        filter_edit_instructions = replace_prefilter(edit_instructions)
    elif args.instruction_type == 'action_change':
        filter_edit_instructions = action_prefilter(edit_instructions)
    else:
        filter_edit_instructions = edit_instructions
    print("after specific rules: ", len(filter_edit_instructions))

    final_instructions = pre_filter(filter_edit_instructions, args.instruction_type, device=device,
                                    image_root=args.image_root, clip_model=clip_model, clip_preprocess=clip_preprocess)
    print("after final implicit filter: ", len(final_instructions))

    if line_mode:
        with open(args.instruction_path.replace(".jsonl", "_filtered.jsonl"), "w") as f:
            json.dump(final_instructions, f, indent=4)
    else:
        with open(args.instruction_path.replace(".jsonl", "_filtered.jsonl"), 'w') as f:
            for instruction in final_instructions:
                f.write(json.dumps(instruction) + '\n')
