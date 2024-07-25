import selfies as sf
import json
from tqdm import tqdm

with open('./description_guided_molecule_design.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

train = []
valid = []
test = []
print(sf.get_semantic_constraints())
for i,info in tqdm(enumerate(data)):
    decoded_smiles = sf.decoder(info['output'],compatible=True)  # SELFIES --> SMILES
    cid = str(i)
    source = info['instruction'] + info['input']
    source = source.replace("\n", "\\n")
    target = decoded_smiles
    print(source,target,info['output'],info)
    if info['metadata']['split'] == 'train':
        train.append((cid,source,target))
    if info['metadata']['split'] == 'valid':
        valid.append((cid,source,target))
    if info['metadata']['split'] == 'test':
        test.append((cid,source,target))

with open('./train.txt','w',encoding='utf-8') as f:
    for i in train:
        f.write(i[0] + '\t' + i[2] + '\t' + i[1] + '\n')
with open('./val.txt','w',encoding='utf-8') as f:
    for i in valid:
        f.write(i[0] + '\t' + i[2] + '\t' + i[1] + '\n')
with open('./test.txt','w',encoding='utf-8') as f:
    for i in test:
        f.write(i[0] + '\t' + i[2] + '\t' + i[1] + '\n')