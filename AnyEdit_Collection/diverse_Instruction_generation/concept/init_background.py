'''
在下面代码的基础上遍历concept_background_case0.json，把它的key中的内容存到concept_pool[concept]['b']下面
，存之前先转为set，这样可以去重
'''

import json

def create_concept_pool(input_file='concept0.json',
                        background_file= 'concept_background_case0.json',
                        output_file='concept_pool.json'):
    '''
    初始化concept_pool，初始化结束后没啥用了
    :param input_file:
    :param background_file:
    :param output_file:
    :return:
    '''
    with open(input_file, 'r', encoding='utf-8') as file:
        concepts = json.load(file)
    with open(background_file, 'r', encoding='utf-8') as file:
        background_data = json.load(file)
    concept_pool = {}
    for concept in concepts:
        concept_pool[concept] = {'b': set(), 'c': ''}

    # 遍历背景知识，添加到对应的概念b中
    for concept, backgrounds in background_data.items():
        if concept in concept_pool:  # 确保概念在concept_pool中
            # 将背景知识转换为小写并去重
            concept_pool[concept]['b'] = set([background.lower() for background in backgrounds])

    # 将集合转换回列表，以便可以json化
    for concept in concept_pool:
        concept_pool[concept]['b'] = list(concept_pool[concept]['b'])

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(concept_pool, file, indent=4, ensure_ascii=False)


def extract_keys_to_file(input_file='concept_pool.json', output_file='concept_pool_key.json'):
    '''
    得到key文件
    :param input_file:
    :param output_file:
    :return:
    '''
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    keys = list(data.keys())
    print(len(keys))
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(keys, file, indent=4, ensure_ascii=False)

extract_keys_to_file()
