# todo: 暂时不支持batch size。之后把gen改一下支持batch后可以把这个文件和instruction_gen.py合并
import json
import random
from tqdm import tqdm
from edit_instruction.prompt_generation_tool import get_content_instruction, instruction_evaluation, few_example_dict, generate_tags
import jsonlines
import argparse
from concept.utils import init_model, text_batch, extract_answer, model_generate
from termcolor import cprint
from edit_instruction.instruction_gen import instruction_type_list, CustomDataset, CustomDataLoader

def gen(description, image_file, special_flag=False):
    '''
    not support batch size
    '''
    parser_results = generate_tags(description)

    if args.instruction_type == 'mix':
        if special_flag:
            temp_list = [instr for instr in instruction_type_list if instr != "tone_transfer"]
            instruction_type = random.choice(temp_list)
        else:
            instruction_type = random.choice(instruction_type_list)
    else:
        instruction_type = args.instruction_type

    if instruction_type in ['color_alter', 'appearance_alter', 'remove', 'replace'] \
            and len(parser_results['nouns']) == 0:
        return None
    if instruction_type in ['action_change'] and len(parser_results['verb']) == 0:
        return None

    prompt = get_content_instruction(new_prompt=description, few_shot_examples=few_example_dict[instruction_type],
                                     instruction_type=instruction_type, tokenizer=tokenizer).to(model.device)
    model_inputs = text_batch(input_ids_list=[prompt], tokenizer=tokenizer)
    generated_ids = model_generate(model, tokenizer, model_inputs)

    instruction_result = extract_answer(generated_ids, tokenizer, model_inputs['input_ids'])[0]

    if instruction_result[-1] != '}' and '}' not in instruction_result:
        instruction_result = instruction_result + '}'
    elif '}' in instruction_result:
        idx = instruction_result.find('}')
        instruction_result = instruction_result[:idx + 1]
    try:
        instruction_result = eval(instruction_result)
    except Exception:
        print('Error instruction: ', instruction_result)
        return None
    instruction_result['input'] = description
    instruction_result['edit_type'] = instruction_type
    instruction_result['image_file'] = image_file

    if 'edit' not in instruction_result.keys():
        return None
    self_prompt = instruction_evaluation(instruction=instruction_result['edit'],
                                         instruction_type=instruction_type, tokenizer=tokenizer).to(model.device)
    self_model_inputs = text_batch(input_ids_list=[self_prompt], tokenizer=tokenizer)
    self_generated_ids = model_generate(model, tokenizer, self_model_inputs)
    eval_answer = extract_answer(self_generated_ids, tokenizer, self_model_inputs['input_ids'])[0]
    if 'yes' in eval_answer.lower():
        return instruction_result

final_results= []
def multi_turn_generation(description, image_file, now_iter, total_iter, histroy_edit=None):
    if now_iter == 1:
        assert histroy_edit is None, 'the history must be empty in first iteration'
        histroy_edit = [{"input": description}]

    result = gen(description=description, image_file=image_file, special_flag=False if now_iter==total_iter else True)

    if result is not None: # 中间有一个没生成就直接跳过
        histroy_edit.append(result)
        if now_iter==total_iter:
            assert len(histroy_edit) == total_iter+1, 'missing some edit instruction'
            final_results.append(histroy_edit)
        else:
            multi_turn_generation(description=description, image_file=image_file,
                                  now_iter=now_iter+1, total_iter=total_iter, histroy_edit=histroy_edit)

def parse_args():
    parser = argparse.ArgumentParser(description="Instruction Pipeline")
    parser.add_argument("--instruction-type", default='remove',
                        choices=['mix'] + instruction_type_list, help="specify the experiment id.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument("--idx", type=int, default=-100, help="specify the experiment id.")
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
    dataset_type = 'mscoco'
    dataloader = CustomDataLoader(CustomDataset('./AnyEdit/AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/image_caption.json', dataset_type), batch_size=1, shuffle=False)

    for caption_data_list in tqdm(dataloader):
        multi_turn_generation(description=caption_data_list[0]['caption'], image_file=caption_data_list[0]['image_file'],
                              now_iter=1, total_iter=args.iter)
        with open(f"./AnyEdit/AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/output/edit_{args.instruction_type}_instructions_{dataset_type}_{args.idx}.jsonl", "w") as f:
            json.dump(final_results, f, indent=4)
