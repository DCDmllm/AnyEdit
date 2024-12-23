'''
Stage 1:
    编码抽好，放在concept_emb.json里面
Stage 2:
    两两计算相似度，存为 3元组的 格式 存在json
Stage 3:
    从高到低排序
Stage 4:
    绘制饼状图，各个数值段的比例

nohup python fliter_concept.py > fliter.log 2>&1
'''

import clip
import torch
import json
import os
from tqdm import tqdm
import numpy as np
import csv
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def stage_first():
    '''
    抽取特征并保存
    '''
    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    with open('concept0.json', 'r') as f:
        concepts = json.load(f)

    progress_bar = tqdm(concepts)
    concept_features = {}

    for concept in progress_bar:
        progress_bar.set_description(f"Processing {concept}")
        text_tokens = clip.tokenize([concept]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        concept_features[concept] = text_features.cpu().numpy().tolist()

    with open('concept_emb0.json', 'w') as f:
        json.dump(concept_features, f)

    print("Completed. The new concept.json file has been saved.")


def stage_second():
    with open('concept_emb0.json', 'r') as f:
        concept_features = json.load(f)

    with open('similarities0.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['concept1', 'concept2', 'similarity'])
        # 使用 tqdm 包裹第一个循环以显示进度
        for concept1, features1 in tqdm(concept_features.items(), desc="Calculating similarities"):
            text_features1 = np.array(features1)
            for concept2, features2 in concept_features.items():
                if concept1 != concept2:
                    text_features2 = np.array(features2)
                    similarity = text_features1 @ text_features2.T
                    similarity_value = similarity[0][0]  # 提取单个相似度值
                    writer.writerow([concept1, concept2, similarity_value])
    print("Completed. The similarities.csv file has been saved.")

def stage_third():
    '''
    排序
    '''
    data = pd.read_csv('similarities0.csv')
    data_sorted = data.sort_values(by='similarity', ascending=False)
    data_sorted.to_csv('similarities_desc0.csv', index=False)

def stage_forth():
    '''
    请为我写一个python程序，读取similarities.csv，其大致的key内容如下    ['concept1', 'concept2', 'similarity']，请为我统计similarity分别在[, 0.5],[0.5,0.6], [0.6,0.7],[0.7,0.8],[0.8,0.9] [0.9, ]
    这些区间内的数量，然后绘制一张饼状图
    '''
    data_sorted = pd.read_csv('similarities_desc0.csv')
    bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>=0.9']
    data_sorted['similarity_bin'] = pd.cut(data_sorted['similarity'], bins=bins, labels=labels, right=False)
    similarity_counts = data_sorted['similarity_bin'].value_counts().sort_index()

    # ----------------------- 下面是测试数据
    # data_sorted = pd.DataFrame({
    #     'similarity': np.random.uniform(0, 1, 1000)  # 生成1000个介于0到1之间的随机数
    # })
    #
    # bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # labels = ['<0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '>=0.9']
    # data_sorted['similarity_bin'] = pd.cut(data_sorted['similarity'], bins=bins, labels=labels, right=False)
    # similarity_counts = data_sorted['similarity_bin'].value_counts().sort_index()

    # ------- --- 南丁格尔玫瑰图
    total = similarity_counts.sum()
    radii = similarity_counts

    # 根据数量比例计算角度
    angles = np.linspace(0, 2 * np.pi, len(radii), endpoint=False)
    proportional_angles = np.cumsum(radii / total * 2 * np.pi)  # 根据比例计算累积角度
    start_angles = np.insert(proportional_angles[:-1], 0, 0)  # 计算每个扇区的起始角度

    # 绘制南丁格尔玫瑰图
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(radii)))  # 使用viridis颜色图，颜色更清晰
    bars = ax.bar(start_angles, radii, width=np.diff(np.append(start_angles, 2 * np.pi)),
                  color=colors, align='edge', edgecolor='white')

    # 在每个扇区外围添加文本注释
    for bar, label, count in zip(bars, labels, radii):
        angle = bar.get_x() + bar.get_width() / 2
        # 计算标签位置距中心的距离，增加的数字用于调整标签的远近
        distance = bar.get_height()*0.7 + max(radii) * 0.7
        # 使用较大的字体大小，并显示绝对数值
        ax.text(angle, distance, f'{label}\n{count//1000} ({count/total*100:.2f}%)', ha='center', va='center', fontsize=8, color='black')

    ax.set_yticklabels([])  # 移除半径上的标签
    ax.set_xticks(start_angles + np.diff(np.append(start_angles, 2 * np.pi)) / 2)  # 设置角度标签的位置
    ax.set_xticklabels([])  # 移除最外圈的标签
    plt.show()
    plt.savefig('Similarity_Distribution_0.pdf')

def stage_fifth():
    with open('concept_emb0.json', 'r') as f:
        concept_features = json.load(f)

    model, preprocess = clip.load("ViT-B/32")
    model.cuda().eval()
    concepts = ['car', 'bird', 'flower']

    for key in concepts:
        text_tokens = clip.tokenize([key]).cuda()
        with torch.no_grad():
            text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        key_feature = text_features.cpu().numpy()

        for concept1, features1 in tqdm(concept_features.items(), desc="Calculating similarities"):
            text_features1 = np.array(features1)
            similarity = text_features1 @ key_feature.T
            similarity_value = similarity[0][0]
            if similarity_value > 0.85:
                print(key , concept1)

stage_first()
stage_second()
stage_third()
stage_forth()