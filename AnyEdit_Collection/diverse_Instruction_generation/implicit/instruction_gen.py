'''
ref to https://github.com/YangLing0818/EditWorld/blob/main/gpt_script/text_img_gen_aigcbest_full.py
'''
import json
from termcolor import cprint
import os
import argparse
from termcolor import cprint
from concept.utils import init_model, text_batch, extract_answer, model_generate

# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def obtain_text(model, tokenizer, messages):
    cprint(messages, 'cyan')
    encodeds = tokenizer.apply_chat_template(
        conversation=messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).squeeze(0).to(model.device)

    model_inputs = text_batch(input_ids_list=[encodeds], tokenizer=tokenizer)

    generated_ids = model_generate(model, tokenizer, model_inputs, max_new_tokens=1000)
    assert generated_ids.shape[0] == 1, 'only support batch size 1'  # 由于要做history所以只能这样了

    answer = extract_answer(generated_ids, tokenizer, model_inputs['input_ids'])[0]
    cprint(answer, 'yellow')

    return answer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--define_json', type=str, default='./implicit/define_samples.json', help="json file of text samples.")
    parser.add_argument('--output_path', type=str, default='./implicit/gen_sample_history/', help="json file of text samples.")
    parser.add_argument('--output_json', type=str, default='text_gen.json', help="json file of text samples.")
    opt = parser.parse_args()

    os.makedirs(opt.output_path, exist_ok=True)

    if os.path.exists(opt.define_json):
        with open(opt.define_json) as f:
            define_dict = json.load(f)
    else:
        print('Error: define_json not exist')

    if os.path.exists(os.path.join(opt.output_path, opt.output_json)):
        with open(os.path.join(opt.output_path, opt.output_json)) as f:
            json_datas = json.load(f)
        i = len(json_datas.keys())
    else:
        json_datas = {}
        i = 0

    model, tokenizer = init_model('llama3')

    for key in define_dict.keys():
        ori_text=define_dict[key]["original_caption"]
        instruct=define_dict[key]["instruction"]
        tar_text=define_dict[key]["target_cation"]
        keywords=define_dict[key]["key_words"]
        init_message = f""" Now you are an "textual prompt creator", Please provide several examples based on real-world physical conditions, \
each example should sequentially include an initial image description, a final image description, image change instructions, and keywords. \
Here's one example: The initial image description is "{ori_text}", the image change instruction is "{instruct}", \
the final image description is "{tar_text}", and the keywords are "{keywords}". Keywords should preferably not be phrases like "paper plane," but rather single words like "apple". \
Please use simple description which is easy for Stable Diffusion model generation.
Please present the examples in the format of "1. {ori_text}; {instruct}; {tar_text}; {keywords}\n2. ...".
"""
        messages = [{"role": "user", "content": init_message}]
        i += 1
        json_datas[f"sample{i}"] = obtain_text(model, tokenizer, messages)
        messages.append({"role": "assistant", "content": json_datas[f"sample{i}"]})

        # 多可以增加多样性，但是会句式单一
        for _ in range(1,2):
            messages.append({"role": "user", "content": 'continue'})
            i += 1
            json_datas[f"sample{i}"] = obtain_text(model, tokenizer, messages)
            messages.append({"role": "assistant", "content": json_datas[f"sample{i}"]})

        with open(os.path.join(opt.output_path, opt.output_json), 'w') as f:
            json.dump(json_datas, f, indent=4)