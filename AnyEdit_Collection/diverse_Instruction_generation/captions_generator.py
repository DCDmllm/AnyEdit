"""
python captions_generator.py --gpu-id 0
"""

import argparse
import json
import os
from termcolor import cprint
import re
import csv
import random
from concept.utils import template_background, template_case
from tqdm import tqdm
from concept.utils import process_text, text_batch, extract_answer, \
    init_model, model_generate, template_caption, process_text_multi_turn, template_caption_background

os.environ['PYTHONPATH'] = './'

class Captions_Generator:
    def __init__(self, args):
        self.args = args
        self.model, self.tokenizer = init_model(args.model, gpu_id=args.gpu_id)

        csv_path = os.path.join(self.args.save_path, f"{self.args.generation_idx}_{self.args.mode}.csv")
        if not os.path.exists(csv_path):
            csv_file = open(csv_path, "a", newline="")
            self.csv_writer = csv.writer(csv_file)
            if args.mode == 'cb2cap':
                self.csv_writer.writerow(["concept", 'background', "caption"])
            elif args.mode == 'cc2cap':
                self.csv_writer.writerow(["concept1", 'concept2', "caption"])
            else:
                self.csv_writer.writerow(["concept", "caption"])
        else:
            csv_file = open(csv_path, "a", newline="")
            self.csv_writer = csv.writer(csv_file)

    def generate_caption(self, concepts):
        save_path = os.path.join(self.args.save_path)
        os.makedirs(save_path, exist_ok=True)

        for start_idx in tqdm(range(0, len(concepts), self.args.batch_size), desc="Generating captions"):
            batch_concepts = concepts[start_idx:start_idx + self.args.batch_size]
            input_ids_list = []
            for idx, concept in enumerate(batch_concepts):
                encodes = process_text(text=template_caption(concept=concept),
                                       tokenizer=self.tokenizer).to(self.model.device)
                input_ids_list.append(encodes)

            model_inputs = text_batch(input_ids_list=input_ids_list, tokenizer=self.tokenizer)

            generated_ids = model_generate(self.model, self.tokenizer, model_inputs, max_new_tokens=50)
            decoded = extract_answer(generated_ids, self.tokenizer, model_inputs['input_ids'])

            for idx, response in enumerate(decoded):
                self.csv_writer.writerow([batch_concepts[idx], response])

    def generate_background(self, concepts):
        save_path = os.path.join(self.args.save_path)
        os.makedirs(save_path, exist_ok=True)
        history = [template_background('sheepdog'), template_case()]
        for start_idx in tqdm(range(0, len(concepts), self.args.batch_size), desc="Generating backgrounds"):
            batch_concepts = concepts[start_idx:start_idx + self.args.batch_size]
            input_ids_list = []
            for idx, concept in enumerate(batch_concepts):
                encodes = process_text_multi_turn(history=history, text=template_background(concept=concept),
                                       tokenizer=self.tokenizer).to(self.model.device)

                input_ids_list.append(encodes)

            model_inputs = text_batch(input_ids_list=input_ids_list, tokenizer=self.tokenizer)
            generated_ids = model_generate(self.model, self.tokenizer, model_inputs, max_new_tokens=1000)
            decoded = extract_answer(generated_ids, self.tokenizer, model_inputs['input_ids'])

            for idx, response in enumerate(decoded):
                self.csv_writer.writerow([batch_concepts[idx], response])

    def generate_captions_c_b(self, concepts): 
        save_path = os.path.join(self.args.save_path)
        os.makedirs(save_path, exist_ok=True)
        keys = list(concepts.keys())

        for idx, concept in tqdm(enumerate(concepts), desc="Generating captions", total=len(concepts)):
            concept = random.choice(keys)
            random_item = random.choice(concepts[concept])
            cprint(f'{concept, random_item}', 'blue')
            encodes = process_text(text=template_caption_background(concept, random_item),
                                   tokenizer=self.tokenizer).to(self.model.device)

            model_inputs = encodes.to(self.distributed_state.device)
            generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True,
                                                pad_token_id=self.tokenizer.eos_token_id)
            decoded = self.tokenizer.batch_decode(generated_ids)
            response = re.search(r'\[/INST\](.*?)</s>', decoded[0], re.DOTALL).group(1).strip() # 给一个case就1
            response = re.sub( r'\s*\([^)]*\)$', '', response) # 生成了之后记得A woman's knee-length garment with a fluid hemline. (11 words)末尾筛掉
            cprint(response,'red')
            self.csv_writer.writerow([concept, random_item, response])

    def generate_captions_c_c(self, concepts, backgrounds):
        save_path = os.path.join(self.args.save_path)
        os.makedirs(save_path, exist_ok=True)

        for start_idx in tqdm(range(0, len(concepts), self.args.batch_size), desc="Generating compositional captions"):
            batch_concepts = concepts[start_idx:start_idx + self.args.batch_size]
            input_ids_list = []
            for idx, concept in enumerate(batch_concepts):
                encodes = process_text(text=template_caption_background(concept[0], concept[1]),
                                       tokenizer=self.tokenizer).to(self.model.device)
                input_ids_list.append(encodes)

            model_inputs = text_batch(input_ids_list=input_ids_list, tokenizer=self.tokenizer)

            generated_ids = model_generate(self.model, self.tokenizer, model_inputs, max_new_tokens=50)
            decoded = extract_answer(generated_ids, self.tokenizer, model_inputs['input_ids'])
            for idx, response in enumerate(decoded):
                if 'on the' in response[-20:] or 'in the' in response[-20:] or 'at the' in response[-20:]:
                    self.csv_writer.writerow([batch_concepts[idx][0], batch_concepts[idx][1], response])
                else:
                    total_backgrounds = random.sample(backgrounds, k=10)
                    for background in total_backgrounds:
                        update_response = response.strip('.') + f' {background}.'
                        self.csv_writer.writerow([batch_concepts[idx][0], batch_concepts[idx][1], update_response])

def main(args):
    rewriter = Captions_Generator(args)

    with open(args.metadata_filepath, "r") as f:
        concepts = json.load(f)
    if args.mode == 'c2cap':
        rewriter.generate_caption(concepts)
    elif args.mode == 'c2b':
        rewriter.generate_background(concepts)
    elif args.mode == 'cb2cap':
        rewriter.generate_captions_c_b(concepts)
    elif args.mode == 'cc2cap':
        total_c = concepts['concepts']
        backgrounds = concepts['backgrounds']
        combination_concepts = []
        for c1 in total_c:
            for c2 in total_c:
                if c1 != c2 and [c1,c2] not in combination_concepts and [c2,c1] not in combination_concepts:
                    combination_concepts.append([c1,c2])
        rewriter.generate_captions_c_c(combination_concepts, backgrounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for synthetic caption generation.")
    parser.add_argument("--save_path", type=str, default="./AnyEdit_Collection/diverse_Instruction_generation/synthetic_instructions")
    parser.add_argument("--metadata_filepath", type=str, default="./AnyEdit_Collection/diverse_Instruction_generation/concept/concept.json")
    parser.add_argument("--generation_idx", type=int, default=0)  
    parser.add_argument("--mode", type=str, default='cc2cap', choices=['c2b', 'c2cap', 'cb2cap', 'cc2cap'])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--model', choices=["mistral", "llama3"], default='llama3')
    parser.add_argument('--gpu-id', type=int, default=4)
    args = parser.parse_args()
    main(args)