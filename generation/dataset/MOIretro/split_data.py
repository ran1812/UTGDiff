import selfies as sf
import json
from tqdm import tqdm

with open('./retrosynthesis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train = []
valid = []
test = []
for i,info in tqdm(enumerate(data)):
    decoded_smiles = sf.decoder(info['output'],compatible=True)  # SELFIES --> SMILES
    decoded_source = sf.decoder(info['input'],compatible=True)  # SELFIES --> SMILES
    cid = str(i)
    source_prompt = info['instruction']
    source_smiles = decoded_source

    target = decoded_smiles
    if info['metadata']['split'] == 'train':
        train.append((cid,source_prompt,source_smiles,target))
    if info['metadata']['split'] == 'valid':
        valid.append((cid,source_prompt,source_smiles,target))
    if info['metadata']['split'] == 'test':
        test.append((cid,source_prompt,source_smiles,target))

with open('./train.txt','w',encoding='utf-8') as f:
    for i in train:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\n')
with open('./val.txt','w',encoding='utf-8') as f:
    for i in valid:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\n')
with open('./test.txt','w',encoding='utf-8') as f:
    for i in test:
        f.write(i[0] + '\t' + i[1] + '\t' + i[2] + '\t' + i[3] + '\n')