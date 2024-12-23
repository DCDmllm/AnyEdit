import json
import random
import re
from tqdm import tqdm
from edit_instruction.prompt_generation_tool import get_content_instruction, instruction_evaluation, few_example_dict, generate_tags
import jsonlines
import argparse
from concept.utils import init_model, text_batch, extract_answer, model_generate
from termcolor import cprint
from torch.utils.data import Dataset, DataLoader

instruction_type_list = ["add", 'remove', "replace", "color_alter",
                         "action_change", "background_change", "tone_transfer",
                         "textual", "appearance_alter"]

class CustomDataset(Dataset):
    def __init__(self, file_path, source_data):
        self.data = []
        if source_data=="mscoco":  # image_caption.json
            coco_train2017_captions = json.load(open(file_path))
            for k in coco_train2017_captions:
                self.data.append({"image_file": coco_train2017_captions[k]['image_file'],
                                  "caption": random.sample(coco_train2017_captions[k]['captions'], 1)[0].lower()})
        elif source_data in ["art", "coco_text", 'icdar2017rctw', 'LSVT', 'mlt2019', 'MTWI2018', 'ReCTS']:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            for entry in json_data['data_list']:
                if 'annotations' not in entry.keys():
                    continue
                if entry['annotations'][0]['language'] == "Latin":
                    self.data.append({
                        "text": entry['annotations'][0]["text"],
                        "caption": entry['caption'],
                        'image_file': entry["img_name"]
                    })
        elif source_data=='llava595':  # new_llava595_1.json
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                         collate_fn=self.custom_collate_fn)

    def custom_collate_fn(self, batch):
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Pipeline")
    parser.add_argument("--instruction-type", default='remove',
                        choices=['mix'] + instruction_type_list, help="specify the experiment id.")
    parser.add_argument("--source-data", default='llava595', choices=['mscoco', 'art', 'llava595', 'coco_text',
                                                                      'icdar2017rctw', 'LSVT', 'mlt2019', 'MTWI2018',
                                                                      'ReCTS'])
    parser.add_argument("--gpu-id", type=int, default=2, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--idx", type=int, default=-101, help="specify the experiment id.")
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

    model, tokenizer = init_model('llama3', args.gpu_id)
    input_descriptions, distributed_ids = [], []

    if args.source_data=='mscoco':
        assert args.instruction_type != "textual"
        input_file = './AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/image_caption.json'
        output_lines = jsonlines.open(f"./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/output/{args.source_data}_{args.instruction_type}_{args.idx}.jsonl", "w")
    elif args.source_data in ["art", "coco_text", 'icdar2017rctw', 'LSVT', 'mlt2019', 'MTWI2018', 'ReCTS']:
        assert args.instruction_type == "textual"
        input_file = f'./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/textual_{args.source_data}.json'
        output_lines = jsonlines.open(f"./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/output/{args.source_data}_{args.instruction_type}_{args.idx}.jsonl", "w")
    elif args.source_data=='llava595':
        assert args.instruction_type != "textual"
        input_file = './AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/new_llava595.json'
        output_lines = jsonlines.open(f"./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/output/{args.source_data}_{args.instruction_type}_{args.idx}.jsonl", "w")

    dataloader = CustomDataLoader(CustomDataset(input_file,source_data=args.source_data),
                                  batch_size=args.batch_size, shuffle=False)

    for caption_data_list in tqdm(dataloader):
        desc_list, prompt_list, instr_type_list, result_list, checked_result_list = [], [], [], [], []
        for caption_data in caption_data_list:
            if args.instruction_type == 'textual':
                description = f"'description': '{caption_data['caption']}', 'text': '{caption_data['text']}'."
            else:
                description = caption_data['caption']
                parser_results = generate_tags(description)

            if args.instruction_type == 'mix':
                instruction_type = random.choice(instruction_type_list)
            else:
                instruction_type = args.instruction_type

            if instruction_type in ['color_alter', 'appearance_alter', 'remove', 'replace'] \
                    and len(parser_results['nouns']) == 0:
                continue
            if instruction_type in ['action_change'] and len(parser_results['verb']) == 0:
                continue
            instr_type_list.append(instruction_type)
            desc_list.append(description)

            prompt = get_content_instruction(new_prompt=description, few_shot_examples=few_example_dict[instruction_type],
                                             instruction_type=instruction_type, tokenizer=tokenizer).to(model.device)
            prompt_list.append(prompt)

        if prompt_list is []:
            continue
        model_inputs = text_batch(input_ids_list=prompt_list, tokenizer=tokenizer)
        generated_ids = model_generate(model, tokenizer, model_inputs)

        answer_list = extract_answer(generated_ids, tokenizer, model_inputs['input_ids'])
        for (instruction_result,  description, instruction_type, caption_data) in \
                zip(answer_list, desc_list, instr_type_list, caption_data_list):
            if instruction_result[-1]!='}' and '}' not in instruction_result:
                instruction_result = instruction_result + '}'
            elif '}' in instruction_result:
                idx = instruction_result.find('}')
                instruction_result = instruction_result[:idx+1]
            try:
                instruction_result = eval(instruction_result)
            except Exception:
                print('Error instruction: ', instruction_result)
                continue

            if args.instruction_type != 'textual':
                instruction_result['input'] = description
            instruction_result['edit_type'] = instruction_type
            instruction_result['image_file'] = caption_data['image_file']
            result_list.append(instruction_result)

        self_prompt_list = []
        for (instruction_result, instruction_type) in zip(result_list, instr_type_list):
            # self evaluation, if edit pipeline needs instruction for editing, like ip2p
            if any(item not in instruction_result.keys() for item in ['input', 'output', 'edit']) or \
                    instruction_result['input'] == instruction_result['output']:
                continue
            self_prompt = instruction_evaluation(instruction=instruction_result['edit'],
                                                 instruction_type=instruction_type, tokenizer=tokenizer).to(model.device)
            self_prompt_list.append(self_prompt)
            checked_result_list.append(instruction_result)

        self_model_inputs = text_batch(input_ids_list=self_prompt_list, tokenizer=tokenizer)
        self_generated_ids = model_generate(model, tokenizer, self_model_inputs)

        assert len(extract_answer(self_generated_ids, tokenizer, self_model_inputs['input_ids'])) == len(checked_result_list), 'len error'
        for (eval_answer, instruction_result) in \
            zip(extract_answer(self_generated_ids, tokenizer, self_model_inputs['input_ids']), checked_result_list):
            if 'yes' in eval_answer.lower():
                if args.instruction_type == "appearance_alter": # double-check remove color change in appearance
                    instr = instruction_result['edit']
                    if not all(word not in instr for word in ['color', 'white', 'black', 'green', 'blue', 'red', 'grey', 'pink']):
                        continue
                if args.instruction_type == "textual":
                    if instruction_result['input'].count("'") != 2: # only want one text in caption
                        continue
                output_lines.write(instruction_result)
