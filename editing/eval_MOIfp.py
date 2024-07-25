from rdkit import Chem
from evaluation import fingerprint, fcd, cal_exact, cal_valid

target_path = './dataset/MOIfp/test.txt'
test_path = './generation_results/MOIfp_15_MBR_(41_83_2).txt'

truth_list = []
with open(target_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split('\t')
        mol =  Chem.MolFromSmiles(line[3])
        truth_list.append(line[3])

test_list = []
with open(test_path,'r', encoding="utf-8") as f:
    for idx, line in enumerate(f.readlines()):
        line = line.split(' ')
        test_list.append(line[0])

test_list = test_list
truth_list = truth_list[:len(test_list)]
print(len(test_list))
cal_exact(test_list,truth_list)

        