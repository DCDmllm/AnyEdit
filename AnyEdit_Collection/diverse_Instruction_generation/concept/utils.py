import torch
import re
from termcolor import cprint
from transformers import AutoModelForCausalLM, AutoTokenizer

def template_caption(concept: str):
    '''
    generate caption for each concept and background
    :param concept:
    :return:
    '''
    return f"""Make up a human-annotated description of an image that contains the following concept: {concept}. 
    Contains no other objects and requires no modifications or descriptions.
    Do not add subjective judgments about the image, it should be as factual as possible. 
    Do not use fluffy, poetic language and avoid using numerical quantities in captions. 
    Output one single grammatically caption that is no longer than 12 words.(It must be done strictly!)"""


def template_caption_background(concept: str, background: str):
    '''
    generate caption for each concept and background
    :param concept:
    :return:
    '''
    return f"""Make up a human-annotated description of an image that contains the following concept: {concept} with {background}. 
    Contains no other objects and requires no modifications or descriptions.
    Do not add subjective judgments about the image, it should be as factual as possible. 
    Do not use fluffy, poetic language and avoid using numerical quantities in captions. 
    Output one single grammatically caption that is no longer than 12 words.(It must be done strictly!)"""

def template_case():
    return """moorland,yard,veterinary clinic,a river,a lake,farmers market,garden,open air market,doggy daycare,mountains,canine sport event,lake,autumn leaves,sledding hill,playground,a dog training camp,pet store,a trail,breed kennel,dog school,snowy field,a field,animal hospital,zoo,paddock,stranger's house,animal shelter,halloween parade,fishing spot,a countryside,mountainside,canine rehabilitation center,park bench,stable,wildlife refuge,backyard,a cityscape,riverbanks,spring blossoms,dog-friendly wineries,sofa,river bank,a dog show,fitness trail,ice fields,dog sitting service,hiking trail,golf course,herding competition,animal rehab center,community center,grasslands,public transportation,mossland,sheep stable,a vehicle,apple orchard,suburbs,animal sanctuary,lakeside,photoshoot studio,a dog race track,beach,academic campus,grooming salon,restoration project,exercise yard,quarry,forest,training school,hills,field,educational farm,dog competition,swimming pool,pond,balloon festival,ski resort,pathway,piers,car,dog kennel,a meadow,vet clinic,canine therapy centers,dog-friendly gym,log cabin,pet adoption fair,jogging trail,trails,countryside,pet adoption center,dog-friendly beach,animal rights protests,community fair,bike path,a road,agriculture show,sports ground,family home,kennel club,city streets,film shooting site,pet daycare,a street,cottage,dog-friendly coffee shop,public transport,harvest field,meadow,a vet clinic,island,a porch,boat dock,winter snow scene,park,apartment,boarding kennels,wildflower meadow,heathland,sunset,seaside,house hallway,state fair,a beach,car rides,parades,sanctuary,dog-friendly beach cabanas,a walking path,animal rights rally,grassland,fruit garden,agility course,a farm,pet charity event,mountain range,a living room,a pet shop,campsite,lighthouse,boat deck,dockland,ranch,flower field,a backyard,dog park,winter landscape,fireplace,pet shop,windy cliff,a forest,ice rink,breeding kennel,dog-friendly holiday resorts,autumn leaves park,daycare center,dog-friendly breweries,pasture,doggy swimming pools,city,hiking track,concert in the park,sunrise,running track,city park,snowy street,escarpment,dog-friendly office spaces,dog-friendly apartment complexes,picnic area,barn,street alley,pet training center,dog show ring,living room,a mountain range,bedroom,a kennel,woodland,a playground,front yard,riversides,countryside inn,nature reserve,hayfield,waterfall,pumpkin patch,dog shelter,children play ground,highlands,dog-friendly bookstores,dog-friendly holiday cabins,breeding center,greenhouse,seashore,marshland,kids party,vineyard,pet-friendly hotel,peek of autumn,country fair,christmas tree farm,town square,water fountain,canine club,soccer field,pet hotel,dog-friendly restaurants,a yard,dog show,valley,stream side,a dog park,plateau,summer greenery,dog day care,farm,waterfalls,lake side,obedience school,animal rescue center,snowy landscape,rainforest,a park,adoption center,obedience class,outdoor cafe,animal boarding,orchard,reality tv show,footpath,pet exhibition hall,rural road,wildlife reserve,dog festival,a barn,foothills,camping site,stables,village green,a pet training school"""

def template_background(concept: str):
    '''
    :param concept:
    :return: generate background for each concept
    '''
    return f"""Please generate some possible backgrounds for the concept {concept}, with each scenario separated by a comma. 
    Make your best effort to ensure diversity and accuracy in the generated results.
    Don't reply with anything else, such as judgments, comments, question, etc."""

# below refer to VILA-1.5
conv_mistral = {
    'roles':("user", "assistant"),
}
conv_llama3 = {
    'roles':("user", "assistant"),
}

def process_text(text, tokenizer, system_prompt=None):
    '''
    输入单个text然后编码
    :param text:
    :param tokenizer:
    :return:
    '''
    if 'mistral' in tokenizer.name_or_path:
        assert system_prompt is None, 'mistral not support system_prompt'
        roles = conv_mistral['roles']
        messages = [
            {"role": roles[0], "content": text}
        ]
        encodeds = tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt"
        )
    elif 'llama' in tokenizer.name_or_path:
        # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
        roles = conv_llama3['roles']
        if system_prompt is None:
            messages = [
                {"role": roles[0], "content": text},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt}, # Example: "You are a pirate chatbot who always responds in pirate speak!"
                {"role": roles[0], "content": text},
            ]
        encodeds = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        raise NotImplementedError

    return encodeds.squeeze(0)

def process_text_multi_turn(history:list, text, tokenizer, system_prompt=None):
    '''
    to encode prompt with history have many chat between user and assistant
    '''
    assert len(history)%2 == 0, 'need have same user and assistant in history'
    if 'mistral' in tokenizer.name_or_path:
        assert system_prompt is None, 'mistral not support system_prompt'
        roles = conv_mistral['roles']

        messages = []
        for idx, his in enumerate(history):
            messages.append({"role": roles[idx%2], "content": his})

        messages.append({"role": roles[0], "content": text})
        encodeds = tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt"
        )
    elif 'llama' in tokenizer.name_or_path:
        # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
        roles = conv_llama3['roles']

        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})  # Example: "You are a pirate chatbot who always responds in pirate speak!"
        for idx, his in enumerate(history):
            messages.append({"role": roles[idx%2], "content": his})
        messages.append({"role": roles[0], "content": text})
        encodeds = tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
    else:
        raise NotImplementedError

    return encodeds.squeeze(0)

def manual_pad(input_ids_list, pad_id):
    '''
    padding 在assitant之前
    :return:
    '''
    assitant_token_len = 4 # llama3, mistral
    max_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids_list = []
    for ids in input_ids_list:
        # Calculate the number of pad tokens to add
        num_padding = max_len - len(ids)
        # Split the ids into two parts: before and after the last 4 tokens, Convert tensor to list
        before_assistant = ids[:-assitant_token_len].tolist()
        assistant_tokens = ids[-assitant_token_len:].tolist()
        # Add padding before the assistant tokens
        padded_ids = before_assistant + [pad_id] * num_padding + assistant_tokens
        padded_input_ids_list.append(torch.tensor(padded_ids, device=ids.device))  # Ensure the tensor is on the same device
    return torch.stack(padded_input_ids_list) # Manually create the [N, max_len] tensor

def text_batch(input_ids_list, tokenizer):
    # mistral and llama3 not define pad token, so using tokenizer.eos_token_id
    # as https://discuss.huggingface.co/t/mistral-trouble-when-fine-tuning-dont-set-pad-token-id-eos-token-id/77928
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/discussions/40
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = manual_pad(input_ids_list, pad_id) # have checked mistral is ok
    # 下列padding可能打在ASSITANT后面可能会有点问题
    # input_ids = torch.nn.utils.rnn.pad_sequence(
    #     input_ids_list,
    #     batch_first=True,
    #     padding_value=pad_id
    # )
    input_ids = input_ids[:, :tokenizer.model_max_length]
    return {'input_ids':input_ids, 'attention_mask':input_ids.ne(pad_id)}

def extract_answer(generated_ids, tokenizer, input_ids):
    response_list = []
    if 'mistral' in tokenizer.name_or_path:
        decoded = tokenizer.batch_decode(generated_ids)
        for coded in decoded:
            response_list.append(re.search(r'\[/INST\](.*?)</s>', coded, re.DOTALL).group(1).strip())
    elif 'llama' in tokenizer.name_or_path:
        for ids in generated_ids:
            response = ids[input_ids.shape[-1]:]
            response_list.append(tokenizer.decode(response, skip_special_tokens=True))
    else:
        raise NotImplementedError
    return response_list

def init_model(type: str, gpu_id=4):
    assert type in ["mistral", "llama3"], "only support mistral and llama3"
    if type == "mistral":
        model_path = './checkpoints/mistral-7b'
    else:
        model_path = './checkpoints/llama3-8b'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=f"cuda:{gpu_id}", torch_dtype=torch.float16)
    return model, tokenizer

def model_generate(model, tokenizer, model_inputs, max_new_tokens=100):
    if 'mistral' in tokenizer.name_or_path:
        generated_ids = model.generate(input_ids=model_inputs['input_ids'],
                                       attention_mask=model_inputs['attention_mask'],
                                       max_new_tokens=max_new_tokens, do_sample=True,
                                       pad_token_id=tokenizer.eos_token_id
                                       )
    else:
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        generated_ids = model.generate(
            input_ids=model_inputs['input_ids'], attention_mask=model_inputs['attention_mask'],
            eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_new_tokens=max_new_tokens
        )

    return generated_ids