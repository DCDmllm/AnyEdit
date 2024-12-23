"""
清洗concept的代码

input: https://github.com/google-research/syn-rep-learn/blob/main/SynCLR/synthesis/syn_text/imgnet21k_combined_background_dict.json
    https://github.com/google-research/syn-rep-learn/blob/main/SynCLR/synthesis/syn_text/imgnet_background_dict.json
output: concept_background_case.json
"""
import json
import os
from tqdm import tqdm
import re

def merge_json(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    merged_data = {}

    # 合并两个字典
    for key in data1:
        merged_data[key] = data1[key]
    for key in data2:
        if key in merged_data:
            merged_data[key].extend(data2[key])
        else:
            merged_data[key] = data2[key]

    print("Number of keys in merged_data[key]:", len(merged_data.keys()))
    with open(f'{output_file}_key.json', 'w', encoding='utf-8') as key_file:
        json.dump(list(merged_data.keys()), key_file, indent=4)

    with open(f'{output_file}.json', 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4)

# concept_background_case.json
# merge_json("imgnet_background_dict.json", "imgnet21k_combined_background_dict.json", "concept_background_case")

def caculatiion():
    """
    classname: 23028
    imagenet21k: 13890
    classname - imagenet21k: 21409
    classname + imagenet21k: 35299
    """
    with open('classname0.json', 'r') as file:
        classname_data = json.load(file)
    classname_keys = set()
    for key in classname_data:
        classname_keys.update(classname_data[key])
    with open('concept_background_case_key0.json', 'r') as file:
        concept_data = json.load(file)
    concept_keys = set(concept_data)
    # 求差集
    difference = classname_keys.difference(concept_keys)

    print(f"classname: {len(classname_keys)}\n"
          f"imagenet21k: {len(concept_keys)}\n"
          f"classname - imagenet21k: {len(difference)}\n"
          f"classname + imagenet21k: {len(difference)+len(concept_keys)}")

    # 将difference并集更新到concept_keys中
    concept_keys.update(difference)
    print(len(concept_keys))
    with open('concept0.json', 'w') as outfile:
        json.dump(list(concept_keys), outfile, indent=4)

def fliter_openimageV7():

    with open('classname.json', 'r') as f:
        data = json.load(f)

    # 过滤掉openimageV7中包含数字或者\u的条目
    filtered_data = []
    for item in data['openimageV7']:
        # 使用encode()函数将字符串转换为字节序列，并指定错误处理方式为忽略
        encoded_item = item.encode('ascii', 'ignore').decode('ascii')
        if not any(char.isdigit() for char in encoded_item):
            filtered_data.append(encoded_item)

    # 将过滤后的数据保存到classname1.json文件
    data['openimageV7'] = filtered_data
    with open('classname.json', 'w') as f:
        json.dump(data, f, indent=4)

def asc_json():
    '''
    json中value按字典序升序排列
    :return:
    '''
    # 读取 classname0.json 文件
    with open('classname0.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 对每个 key 的值进行排序
    for key in data:
        if isinstance(data[key], list):
            data[key].sort()

    # 写入排序后的内容到 classname1.json 文件
    with open('classname1.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def clean_imagenet():
    '''
    :return: concept_background_case0.json
    '''
    with open('concept_background_case0.json', 'r') as file:
        data = json.load(file)

    merged_categories = {}
    # keys_storage = {}
    # 预处理关键字：去除可能的前导空格，并转为小写
    merge_keywords = [keyword.strip().lower() for keyword in [
        'shark', 'turtle', 'lizard', 'Raven', 'Sparrow', 'Swallow', 'Woodpecker', 'Warbler', 'Tern', 'Wren', 'wrench','Waterthrush', 'Vireo',
        'Oriole', 'Kingfisher', 'Hummingbird', 'Gull', 'Flycatcher', 'Cuckoo', 'Cormorant', 'Auklet',
        'Albatross', 'terrier', 'cheese', 'frog', 'dipper', 'salamander', 'newt', 'bullfrog', "gecko",
        "iguana", 'owl','alligator', 'snake', 'python', 'rattlesnake', 'spider', 'grouse', 'parrot'
        "anole", 'cockatoo', 'merganser', 'crab', 'gallinule', 'redshank', 'whale', 'hound', 'Coonhound', 'thistle',
        'lettuce', 'carrot', 'shrike', 'rhubarb', 'weasel', 'locust', 'robin', 'constrictor', 'Blackbird', 'viper', 'parrot',
        'sorghum', 'plum', 'hazel', 'shrew', 'larch', 'marten', 'oil palm', 'spaniel', 'ouzel', 'feverfew',
        'bittern', 'mink', 'white pine', 'holly', 'chestnut', 'squirrel', 'chameleon', 'germander', 'hackberry', 'redstart',
        'spikenard', 'woodcock', 'angelica tree', 'birch', 'barberry', 'sea lion', 'arborvitae', 'raspberry', 'cranberry',
        'oak', 'egret', 'widgeon', 'mistletoe', 'persimmon', 'copper', 'mouse', 'hornbeam', 'honeysuckle', 'badger', 'ash',
        'organ', 'sycamore', 'toad', 'bison', 'crayfish', 'fern', 'cockroach', 'agave', 'parasol', 'crow', 'quaking aspen',
        'foxhound', 'magpie', 'wistaria', 'basswood', 'bulldog', 'Goldfinch', 'Pipit', 'lobster', 'coot', 'bear', 'elm', 'Setter',
        'Sheepdog', 'Sennenhund', 'Corgi', 'Poodle', 'wolf', 'fox', 'beetle', 'butterfly', 'rabbit', 'buffalo', 'camel', 'squash',
        'penstemon', 'missile', 'maple', 'strawberry', 'chipmunk', 'bobtail', 'macaque', 'cloak', 'flatfish', 'tanager', 'woodsia',
        'endive', 'bream', 'goatsucker', 'wildcat', 'dogtooth', 'dewberry', 'dogwood', 'asphodel', 'silver fir', 'polecat',
        'perch', 'curlew', 'nuthatch', 'creeper', 'house cricket', 'white lily', 'sandpipe', 'daisy', 'lemming', 'stingray',
        'truck', 'buckthorn', 'Jay', 'Jaeger', 'Grebe', 'Cowbird', 'catbird', 'Kingbird','Grosbeak', 'Fulmar', 'catfish',
        'dolphin', 'Waxwing', 'Pelican', 'Kittiwake', 'Towhee', 'Shorthair', 'bee'
    ]]

    for key, values in tqdm(data.items(), desc="Merging categories"):
        # 检查关键字是否在键中出现，忽略大小写，保留前导空格
        match_keyword = next((keyword for keyword in merge_keywords if (' ' + keyword + ' ' in key.lower() or ' ' + keyword == key.lower()[-len(keyword)-1:] or keyword == key.lower())), None)
        if match_keyword:
            category = ' '.join(word.capitalize() for word in match_keyword.split())
            merged_categories.setdefault(category, []).extend(values)
            # keys_storage.setdefault(category, []).append(key)
        else:
            merged_categories[key] = values
            # keys_storage[key] = [key]  # 存储原始key

    print(len(list(merged_categories.keys())))
    # 保存合并后的字典
    with open('concept_background_case0.json', 'w') as f:
        json.dump(merged_categories, f, indent=4)
    with open('concept_background_case_key0.json', 'w') as f:
        # json.dump(keys_storage, f, indent=4)
        json.dump(list(merged_categories.keys()), f, indent=4)


def translation():
    '''
    翻译concept
    :return:
    '''
    import requests
    import time
    import hashlib
    import uuid
    def translate_youdao(text, dest_language='zh-CHS'):
        appKey = '5d757c5287fbc7f3'
        appSecret = '5VpPDDREJjDOt7nrFIXpMPzxzjaOrAMA'
        url = "https://openapi.youdao.com/api"
        salt = str(uuid.uuid1())
        sign = hashlib.md5((appKey + text + salt + appSecret).encode('utf-8')).hexdigest()

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'q': text,
            'from': 'auto',
            'to': dest_language,
            'appKey': appKey,
            'salt': salt,
            'sign': sign
        }

        response = requests.post(url, params=payload, headers=headers)
        result = response.json()

        if result['errorCode'] == '0':
            return result['translation'][0]
        else:
            return "Error : " + result['errorCode']

    with open('concept0.json', 'r', encoding='utf-8') as file:
        concepts = json.load(file)
    translated_concepts = {}

    for concept in tqdm(concepts, desc="Translating concepts"):
        translated_concepts[concept] = translate_youdao(concept)
        time.sleep(1)
        print(concept, translated_concepts[concept])

    with open('concept_translate.json', 'w', encoding='utf-8') as file:
        json.dump(translated_concepts, file, ensure_ascii=False, indent=4)

# translation()

with open('concept_pool.json', 'r') as file:
    data = json.load(file)
print(len(data))