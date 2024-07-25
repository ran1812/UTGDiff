from rdkit import Chem
from evaluation import fingerprint, fcd, cal_exact, cal_valid

target_path = './dataset/CHEBI/test.txt'
train_path = './dataset/CHEBI/train.txt'
test_path = './generation_results/MOIfp_15_MBR_(41_83).txt'

truth_list = []
with open(target_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split('\t')
        mol =  Chem.MolFromSmiles(line[1])
        if mol.GetNumAtoms() > 125:
            continue
        truth_list.append(line[1])

train_list = []
with open(train_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split('\t')
        mol =  Chem.MolFromSmiles(line[1])
        if mol.GetNumAtoms() > 125:
            continue
        train_list.append(line[1])

test_list = []
with open(test_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split(' ')
        for i in range(3):
            if 'wrong' in line[i] and i < 2:
                continue
            if Chem.MolFromSmiles(line[i]) is None and i < 2:
                continue
            test_list.append(line[i])
            break

truth_list = truth_list[:len(test_list)]

fingerprint(test_list,truth_list,'RDK')

        