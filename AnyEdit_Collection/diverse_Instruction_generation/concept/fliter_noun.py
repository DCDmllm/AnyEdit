'''
英语
去除人名的名词
小写
nohup python fliter_noun.py > fliter.log 2>&1
Input: https://github.com/hammoudhasan/SynthCLIP/blob/main/TextGen/metadata.json
Output: metadata_flitered.json
'''
import json
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words as nltk_words
from tqdm import tqdm
from termcolor import cprint

# nltk.download('words') # TODO: 第一次启动需要下载
# 加载Spacy英语模型
nlp = spacy.load("en_core_web_sm")
# 初始化WordNetLemmatizer用于词形还原
lemmatizer = WordNetLemmatizer()
# 加载NLTK的英语单词列表进行过滤
english_vocab = set(w.lower() for w in nltk_words.words())

def is_english_word(word):
    # 检查单词是否是英语单词
    return word.lower() in english_vocab

def filter_and_stem_concepts(file_path):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    all_strings = [value for value in data if isinstance(value, str)]

    # 使用Spacy过滤名词并进行词形还原，同时确保词汇是英语
    filtered_concepts = []
    for text in tqdm(all_strings, desc="Processing concepts"):
        doc = nlp(text)
        for token in doc:
            # 过滤非人名的名词
            if token.pos_ in ['NOUN', 'PROPN'] and token.ent_type_ != 'PERSON':
                # 检查是否是英语单词
                if is_english_word(token.text):
                    # 词形还原并添加到结果列表
                    lemma_word = lemmatizer.lemmatize(token.text).lower()
                    filtered_concepts.append(lemma_word)

    unique_concepts = list(set(filtered_concepts)) # 去重
    return unique_concepts

if __name__ == "__main__":
    file_path = 'metadata.json'
    concepts = filter_and_stem_concepts(file_path)
    print(concepts)
    filtered_file_path = 'metadata_filtered.json'
    with open(filtered_file_path, 'w', encoding='utf-8') as file:
        json.dump(concepts, file, ensure_ascii=False, indent=4)