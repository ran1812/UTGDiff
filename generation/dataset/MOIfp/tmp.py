import json

# 定义 JSON 文件的路径
file_path = 'data.json'

# 打开并读取 JSON 文件
with open('./forward_reaction_prediction.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

instance = []
tot = 0
for i in data:
    if i['metadata']['split'] != "train" or tot > 100:
        continue
    instance.append({"input": i['instruction'] + "<bom>" + i['input'] + "<eom>",
                     "output":["<bom>"+i['output']+"<eom>"]})
    tot +=1
data_new = {"Instances": instance, "Definition": [""], "Source":[""]}

with open('forward_reaction_prediction_valid.json', 'w', encoding='utf-8') as file:
    json.dump(data_new, file, ensure_ascii=False, indent=4)