import selfies as sf
import json
from tqdm import tqdm

with open('./reagent_prediction.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train = []
valid = []
test = []
for i,info in tqdm(enumerate(data)):
    source1,source2 = info['input'].split('>>')
    decoded_smiles = sf.decoder(info['output'],compatible=True)  # SELFIES --> SMILES
    decoded_source1 = sf.decoder(source1,compatible=True)  # SELFIES --> SMILES
    decoded_source2 = sf.decoder(source2,compatible=True)  # SELFIES --> SMILES
    cid = str(i)
    source_prompt = info['instruction']


    target = decoded_smiles
    if info['metadata']['split'] == 'train':
        train.append((cid,source_prompt,decoded_source1, decoded_source2,target))
    if info['metadata']['split'] == 'valid':
        valid.append((cid,source_prompt,decoded_source1, decoded_source2,target))
    if info['metadata']['split'] == 'test':
        test.append((cid,source_prompt,decoded_source1, decoded_source2,target))

with open('./train.txt','w',encoding='utf-8') as f:
    for i in train:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[4] + '\n')
with open('./val.txt','w',encoding='utf-8') as f:
    for i in valid:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[4] + '\n')
with open('./test.txt','w',encoding='utf-8') as f:
    for i in test:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\t' + i[4] + '\n')