from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import Chem

with open('./generation_results/CHEBI_15_MBR_(49_79).txt') as f:
    text = f.readlines()

lst = []

for k,i in enumerate(text):
    a = i.split(' ')[0]
    smiles = a

    mol =  Chem.MolFromSmiles(smiles)
    if mol is not None:
        lst.append(smiles)
        continue
    flag = 1
    tot = 0
    while flag == 1 and tot < 5:
        tot += 1
        flag = 0
        mol =  Chem.MolFromSmiles(smiles,False)
        if mol is None:
            s = smiles
            break
        aromatic_atoms = mol.GetAromaticAtoms()
        for atom in aromatic_atoms:
            sym = atom.GetSymbol()
            tmp = Chem.MolFromSmiles(sym)
            degree = atom.GetDegree()  + atom.GetNumExplicitHs()
            
            if degree < tmp.GetAtomWithIdx(0).GetTotalValence() and sym == 'N':
                atom.SetNumExplicitHs(1)
                flag = 1
                break
            atom_idx = atom.GetIdx()
        s = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(s) is not None:
            break
    if Chem.MolFromSmiles(s) is not None:
        lst.append(s)
    else:
        lst.append(smiles)

with open('./generation_results/CHEBI_15_MBR_(49_79)_final_1.txt','w') as f:
    for i in lst:
        f.write(i+'\n')