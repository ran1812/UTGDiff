from rdkit import Chem
from evaluation import fingerprint, fcd, cal_exact, cal_valid, cal_novel, cal_unique
import numpy as np

target_path = './dataset/CHEBI/test.txt'
test_path = './generation_results/CHEBI_15_MBR_(49_79_1)_final.txt'

truth_list = []
with open(target_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split('\t')
        mol =  Chem.MolFromSmiles(line[1])
        num_atoms = mol.GetNumAtoms()
        
        truth_list.append(line[1])
''''''
test_list = []
with open(test_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split(' ')[0]
        test_list.append(line)


test_list = test_list[:len(test_list)]
truth_list = truth_list[:len(test_list)]


ans = fingerprint(test_list,truth_list,'MACCS')
print(ans)