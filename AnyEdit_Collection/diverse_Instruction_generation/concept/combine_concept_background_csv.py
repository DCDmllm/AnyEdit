'''
input: concept_pool.json
        background.csv from background generate
'''
import pandas as pd
import json
from termcolor import cprint

base_file = '../synthetic'
csv_files = ["1_background.csv", "7_background.csv", '6_background.csv'] # todo: 2，5_background.csv坏了. 最后没有读取
dfs = []
for file in csv_files:
    df = pd.read_csv(f'{base_file}/{file}', delimiter='\t')  # 使用制表符分隔字段
    df[['concept', 'caption']] = df['concept,caption'].str.split(',', n=1, expand=True)  # 将一列拆分为两列
    df.drop(columns=['concept,caption'], inplace=True)  # 删除原始的合并列
    dfs.append(df)
merged_df = pd.concat(dfs)

print(f'[Start]  df length is {len(merged_df)}')
merged_df = merged_df.dropna(subset=['caption']) # 过滤掉caption为空的行
merged_df['caption'] = merged_df['caption'].apply(lambda x: x.strip('""').split(','))
merged_df = merged_df[merged_df['caption'].apply(len) > 1] # 过滤掉caption为空的行
merged_df = merged_df[~merged_df['concept'].str.contains('note|please|Note|Please|1.|2.|3.|4.|5.|6.|7.|8.|9.|0.|\*', case=False)] # 过滤掉一些奇怪的结果
print(f'[Processing]  df length is {len(merged_df)}')
# 已经检查过，可以合并list中值; 如果value的list长度太小说明也可能是某些错误字符
merged_dict = {key: value for key, value in merged_df.groupby('concept')['caption'].apply(lambda x: list(set(y for sublist in x for y in sublist))).to_dict().items() if len(value) >= 5}
print(f'[Processing]  df length is {len(merged_dict)}')

with open('concept_pool.json', 'r', encoding='utf-8') as f:
    concept_pool = json.load(f)

filtered_merged_dict = {}
for key, value in merged_dict.items():
    if key in concept_pool:
        filtered_merged_dict[key] = value + concept_pool[key]['b']

print(f'[End]  after filter with concept pool the df length is {len(filtered_merged_dict)}')

for key, value in filtered_merged_dict.items():
    filtered_merged_dict[key] = list(set(item.strip() for item in value))

with open("merged_concept_background.json", "w") as json_file:
    json.dump(filtered_merged_dict, json_file, indent=4)


