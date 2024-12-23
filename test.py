import json

with open('/home1/yqf/ssd/final_output/mscoco_action_change_-100.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
print(len(data))