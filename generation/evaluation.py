import torch
from rdkit import Chem
from rdkit.Chem import Draw,AllChem,MACCSkeys

import numpy as np
import sys
from rdkit import DataStructs
from fcd import get_fcd, load_ref_model, canonical_smiles

def graph2smile_new(tokens, edges, tokenizer,start_value=50265,end_value=50266):
    smiles_batch = []
    for node, edge in zip(tokens, edges):   
        flag=0    
        mol = Chem.RWMol()
        start_indices = (node == start_value).nonzero()[:, 0]
        end_indices = (node == end_value).nonzero()[:, 0]
        if start_indices.shape[0] != 1 or end_indices.shape[0] != 1:
            smiles_batch.append('start_or_end_wrong')
            continue
        node_dict=  {}
        tot=0
        for i,node_id in enumerate(node[start_indices+1:]):
            if node_id > 50266:
                node_str = tokenizer.decode(node_id)[1:-1]
                node_dict[i] = tot
                tot += 1
            else:
                continue
            if (node_str[-1] == '+' or node_str[-1] == '-'):
                atom_name = node_str[:-1]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-1:] + '1'))
            elif len(node_str) > 2 and (node_str[-2] == '+' or node_str[-2] == '-'):
                atom_name = node_str[:-2]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-2:]))
            else:
                atom_name = node_str
                atom = Chem.Atom(atom_name)
            mol.AddAtom(atom)
        if(flag):
            continue
        
        for i in range(1,5):
            start,end = torch.where(edge == i)
            for atom1, atom2 in zip(start,end):
                if atom1.item() not in node_dict or atom2.item() not in node_dict:
                    flag = 1
                    smiles_batch.append('edge_index_wrong')
                    break
                atom1 = node_dict[atom1.item()]
                atom2 = node_dict[atom2.item()]
                if (mol.GetBondBetweenAtoms(atom1, atom2) is None) and atom2 != atom1:
                    if i == 4:
                        mol.AddBond(atom1, atom2, Chem.BondType(Chem.BondType.AROMATIC))
                    else:
                        mol.AddBond(atom1, atom2, Chem.BondType(i))
            if(flag):
                break
        if(flag):
            continue
        smiles = Chem.MolToSmiles(mol)
        smiles_batch.append(smiles)
    return smiles_batch

def graph2smile_newpos(tokens, edges, positions, tokenizer,start_value=50265,end_value=50266):
    smiles_batch = []
    for node, edge, pos in zip(tokens, edges, positions):   
        flag=0    
        mol = Chem.RWMol()
        start_indices = (node == start_value).nonzero()[:, 0]
        end_indices = (node == end_value).nonzero()[:, 0]
        if start_indices.shape[0] == 0 or end_indices.shape[0] == 0:
            smiles_batch.append('start_or_end_wrong')
            print(node,pos)
            continue
        start_indices = start_indices[pos[start_indices]==2]
        end_indices = end_indices[torch.argmin(pos[end_indices])].unsqueeze(0)
        if start_indices.shape[0] != 1 or end_indices.shape[0] != 1:
            smiles_batch.append('start_or_end_wrong')
            print(node,pos)
            continue
        node_dict=  {}
        tot=0
        end_indices = pos[end_indices]

        for i,node_id in enumerate(node[start_indices+1:]):
            if node_id > 50266 and pos[start_indices+1:][i] < end_indices:
                node_str = tokenizer.decode(node_id)[1:-1]
                node_dict[i] = tot
                tot += 1
            else:
                continue
            if (node_str[-1] == '+' or node_str[-1] == '-'):
                atom_name = node_str[:-1]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-1:] + '1'))
            elif len(node_str) > 2 and (node_str[-2] == '+' or node_str[-2] == '-'):
                atom_name = node_str[:-2]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-2:]))
            else:
                atom_name = node_str
                atom = Chem.Atom(atom_name)
            mol.AddAtom(atom)
        if(flag):
            continue
        
        for i in range(1,5):
            start,end = torch.where(edge == i)
            for atom1, atom2 in zip(start,end):
                if atom1.item() not in node_dict or atom2.item() not in node_dict:
                    flag = 1
                    smiles_batch.append('edge_index_wrong')
                    break
                atom1 = node_dict[atom1.item()]
                atom2 = node_dict[atom2.item()]
                if (mol.GetBondBetweenAtoms(atom1, atom2) is None) and atom2 != atom1:
                    if i == 4:
                        mol.AddBond(atom1, atom2, Chem.BondType(Chem.BondType.AROMATIC))
                    else:
                        mol.AddBond(atom1, atom2, Chem.BondType(i))
            if(flag):
                break
        if(flag):
            continue
        smiles = Chem.MolToSmiles(mol)
        smiles_batch.append(smiles)
    return smiles_batch

def graph2smile(tokens, edges, tokenizer,start_value=50265,end_value=50266):
    smiles_batch = []
    for node, edge in zip(tokens, edges):   
        flag=0    
        mol = Chem.RWMol()
        start_indices = (node == start_value).nonzero()[:, 0]
        end_indices = (node == end_value).nonzero()[:, 0]
        if start_indices.shape[0] != 1 or end_indices.shape[0] != 1:
            smiles_batch.append('start_or_end_wrong')
            continue
        for node_id in node[start_indices+1:end_indices]:
            node_str = tokenizer.decode(node_id)[1:-1]
            if node_id < start_value:
                smiles_batch.append('id_wrong')
                flag = 1
                break
            if (node_str[-1] == '+' or node_str[-1] == '-'):
                atom_name = node_str[:-1]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-1:] + '1'))
            elif len(node_str) > 2 and (node_str[-2] == '+' or node_str[-2] == '-'):
                atom_name = node_str[:-2]
                atom = Chem.Atom(atom_name)
                atom.SetFormalCharge(int(node_str[-2:]))
            else:
                atom_name = node_str
                atom = Chem.Atom(atom_name)
            mol.AddAtom(atom)
        mol.UpdatePropertyCache()
        if(flag):
            continue
       
        for i in range(1,5):
            start,end = torch.where(edge == i)
            for atom1, atom2 in zip(start,end):
                atom1 = atom1.item()
                atom2 = atom2.item()
                if atom1 >= end_indices - start_indices - 1 or atom2 >= end_indices - start_indices - 1:
                    flag = 1
                    smiles_batch.append('edge_range_wrong')
                    break
                if (mol.GetBondBetweenAtoms(atom1, atom2) is None) and atom2 != atom1:
                    if i == 4:
                        mol.AddBond(atom1, atom2, Chem.BondType(Chem.BondType.AROMATIC))
                    else:
                        mol.AddBond(atom1, atom2, Chem.BondType(i))
            if(flag):
                break
        if(flag):
            continue
        smiles = Chem.MolToSmiles(mol)

        smiles_batch.append(smiles)
    return smiles_batch

def draw(smiles:str):
    mol = Chem.MolFromSmiles(smiles)
    print(mol)
    if mol is not None:
        # 生成分子图像
        img = Draw.MolToImage(mol)
        img.save('./show.jpg')
    else:
        print("无法解析SMILES字符串")

def cal_valid(smiles:list):
    wrong = 0
    tot = len(smiles)
    valid = []
    for i in range(tot):
        mol = Chem.MolFromSmiles(smiles[i])
        if mol is None:
            wrong +=1
        else:
            valid.append(smiles[i])
    print(1-wrong/tot)
    return valid, 1-wrong/tot

def cal_unique(smiles:list):
    valid,_ = cal_valid(smiles)
    print(len(set(valid)) / len(valid))
    return list(set(valid)), len(set(valid)) / len(valid)

def cal_novel(smiles:list,train_list:list):
    unique,_ = cal_unique(smiles)
    novel = []
    num_novel = 0
    for smile in unique:
        if smile not in train_list:
            novel.append(smile)
            num_novel += 1
        else:
            print(smile)
    print(num_novel / len(unique))
    return novel, num_novel / len(unique)


def cal_exact(smiles1:list, smiles2:list):
    num_exact = 0
    for mol1,mol2 in zip(smiles1, smiles2):
        mol1 = Chem.MolFromSmiles(mol1)
        mol2 = Chem.MolFromSmiles(mol2)
        for atom in mol2.GetAtoms():
            if atom.HasProp('_CIPCode'):
                atom.ClearProp('_CIPCode')
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol2.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        if mol1 is None or mol2 is None:
            continue
        if Chem.MolToInchi(mol1) == Chem.MolToInchi(mol2): 
            num_exact += 1
    print(num_exact/len(smiles1))
    return num_exact/len(smiles1)

def fingerprint(smiles1:list, smiles2:list, name:str):
    fp_list = []
    wrong = 0
    fp_save = []
    for mol1,mol2 in zip(smiles1,smiles2):
        ori1 = mol1
        ori2 = mol2
        flag = 0
        mol1 = Chem.MolFromSmiles(mol1)
        mol2 = Chem.MolFromSmiles(mol2)
        if mol1 is None or mol2 is None:
            wrong +=1
            fp_save.append('wrong')
            print(' ')
            continue

        if name == 'RDK':
            fps1 = Chem.RDKFingerprint(mol1)
            fps2 = Chem.RDKFingerprint(mol2)
            x = DataStructs.FingerprintSimilarity(fps1,fps2)
        elif name == 'MACCS':
            fps1 = MACCSkeys.GenMACCSKeys(mol1)
            fps2 = MACCSkeys.GenMACCSKeys(mol2)
            x = DataStructs.FingerprintSimilarity(fps1,fps2)
        elif name == 'Morgan':
            fps1 = AllChem.GetMorganFingerprint(mol1,2)
            fps2 = AllChem.GetMorganFingerprint(mol2,2)
            x = DataStructs.TanimotoSimilarity(fps1,fps2)            

        fp_save.append(x)
        fp_list.append(x)
    with open('fp_val.txt','w') as f:
        for i in fp_save:
            f.write(str(i)+'\n')
    return np.array(fp_list).mean()



def fcd(smiles1:list, smiles2:list):
    model = load_ref_model()
    smiles3 = []
    for i in smiles2:
        mol2 = Chem.MolFromSmiles(i)
        for atom in mol2.GetAtoms():
            if atom.HasProp('_CIPCode'):
                atom.ClearProp('_CIPCode')
            atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        for bond in mol2.GetBonds():
            bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
        smiles3.append(Chem.MolToSmiles(mol2))

    canon_gt_smis = [w for w in canonical_smiles(smiles3) if w is not None]
    canon_ot_smis = [w for w in canonical_smiles(smiles1) if w is not None]

    fcd_sim_score = get_fcd(canon_gt_smis, canon_ot_smis, model)
    
    print(fcd_sim_score)
    return fcd_sim_score