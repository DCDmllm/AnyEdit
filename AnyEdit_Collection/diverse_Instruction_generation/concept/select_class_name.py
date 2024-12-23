import json

with open('concept_pool.json', 'r') as file:
    data = json.load(file)

keys_to_remove = []
last_str = {}
for key in data.keys():
    if ' ' in key:
        last_word = key.split(' ')[-1]
        if last_word in last_str.keys():
            last_str[last_word] += 1
        else:
            last_str[last_word] = 1

for key in last_str.keys():
    if key in data.keys():
        last_str[key] += 1

sorted_last_str = sorted(last_str.items(), key=lambda x: x[1])
print(dict(sorted_last_str))
print(len(data), len(sorted_last_str))

# for key in data.keys():
#     if key.split(' ')[-1] in last_str.keys() and last_str[key.split(' ')[-1]] > 1 and key.split(' ')[-1] != key:
#         keys_to_remove.append(key)
#
# for key in keys_to_remove:
#     print(key)
#     data.pop(key)
#
# print(len(data))
# with open('concept_pool.json', 'w') as file:
#     json.dump(data, file, ensure_ascii=False, indent=4)
