'''
python gpt_fliter_concept_pool.py
'''
from termcolor import cprint
import openai
import json
from tqdm import tqdm

def gpt_answering(key):
    openai.api_key = 'sk-proj-xxxxx'  # 这里需要替换为你的OpenAI API密钥

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
      You are an expert in determining whether a concept is a good one. 
      A good concept should not be a proper noun (such as a specific bird species, car model, or airplane model), and you should fully understand its meaning. 
      Additionally, a good concept should be something that can be visually represented as a tangible object.
      If you believe the concept is a good one, please respond with "Yes." Otherwise, respond with "No."
            """},
            {"role": "user", "content": """Winter Wren bird"""},
            {"role": "assistant", "content": """No."""},

            {"role": "user", "content": """kelp"""},
            {"role": "assistant", "content": """Yes."""},

            {"role": "user", "content": """Ford Motor"""},
            {"role": "assistant", "content": """No."""},

            {"role": "user", "content": """race"""},
            {"role": "assistant", "content": """No."""},

            {"role": "user", "content": key},
        ],
        # max_tokens=2500
    )

    generated_text = response['choices'][0]['message']['content'] # all content
    gpt_answer = generated_text.split('<answer>')[-1].split('</answer>')[0]

    return gpt_answer

def main():
    with open(fr"concept_pool.json", 'r') as fs:
        data = json.load(fs)
    print(len(data))
    processed = {}
    try:
        for key, value in tqdm(data.items()):
            gpt_answer = gpt_answering(key)
            print(key, gpt_answer)
            if 'y' in gpt_answer.lower():
                processed[key] = value

        with open(fr"concept_pool_new.json", 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)
    except:
        with open(fr"concept_pool_new.json", 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()