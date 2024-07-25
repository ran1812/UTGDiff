import datasets
import os
from functools import partial
import torch
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

charge_num2str = {-5:'-5',-4:'-4',-3:'-3',-2:'-2',-2:'-2',-1:'-',0:'',1:'+',2:'+2',3:'+3',4:'+4',5:'+5',6:'+6',7:'+7',8:'+8'}

class DiffusionLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load(self, task_name, split):
        dataset = datasets.load_dataset('lm1b', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, remove_columns='text')
        return dataset

    def my_load(self, task_name, splits):
        return [self._load(task_name, name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        input_encodings = tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, add_special_tokens=False)
        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
        }

        return encodings

class ConditionalLoader:
    def __init__(self, tokenizer, return_source_length=False):
        self.tokenizer = tokenizer
        self.return_source_length = return_source_length
        self.data_dir = './conditional_data'

    @staticmethod
    def _convert_to_features_original(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=256, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=256, truncation=True, add_special_tokens=False)
        return {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

    def load_original(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        dataset = dataset.map(partial(self._convert_to_features_original, tokenizer=self.tokenizer), batched=True)#, load_from_cache_file=False)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def _load(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        if self.return_source_length:
            dataset = dataset.map(partial(self.add_original_src_length, tokenizer=self.tokenizer))
        dataset = dataset.map(self.add_prompt, load_from_cache_file = False, keep_in_memory = True)
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file = False, keep_in_memory = True)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def add_original_src_length(self, example, tokenizer):
        return {
            'original_src_length': len(tokenizer.encode(example['src'], max_length=256, truncation=True, add_special_tokens=False))
        }

    def my_load(self, splits):
        return [self._load(name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=64, truncation=True, add_special_tokens=False,padding='max_length')
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=64, truncation=True, add_special_tokens=False,padding='max_length')
        encodings = {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

        return encodings

    @staticmethod
    def collate_fn(batch_input, tokenizer):

        input_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source'] + d['target'] + [tokenizer.sep_token_id]
        ) for d in batch_input], batch_first=True)

        attention_mask = torch.ones_like(input_ids)
        target_mask = torch.stack([torch.cat([
            torch.zeros(len(d['source']) + 1), torch.ones(input_ids.size(1) - len(d['source']) - 1)
        ]) for d in batch_input])

        assert input_ids.size() == attention_mask.size() == target_mask.size()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask,
        }

class GraphLoader:
    def __init__(self, tokenizer, return_source_length=False):
        self.tokenizer = tokenizer
        self.return_source_length = return_source_length
        self.data_dir = './dataset'

    @staticmethod
    def _convert_to_features_original(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=512, truncation=True, add_special_tokens=False)
        q2 = tokenizer.batch_encode_plus(example_batch['trg'], max_length=512, truncation=True, add_special_tokens=False)
        return {
            'source': q1['input_ids'],
            'target': q2['input_ids'],
        }

    def load_original(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        dataset = dataset.map(partial(self._convert_to_features_original, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        print(f'Example in {split} set:')
        print(dataset[0])
        return dataset

    def _load(self, split):
        dataset = datasets.load_dataset(os.path.join(self.data_dir, self.task_name, f'{self.task_name}.py'), split=split)
        if self.return_source_length:
            dataset = dataset.map(partial(self.add_original_src_length, tokenizer=self.tokenizer))
        dataset = dataset.map(self.add_prompt)#, load_from_cache_file = False, keep_in_memory = True)
        dataset = dataset.map(partial(self.convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file = False)
        if len(dataset)>0:
            print(dataset)
            print(f'Example in {split} set:')
            print(dataset[0])
        return dataset

    def add_original_src_length(self, example, tokenizer):
        return {
            'original_src_length': len(tokenizer.encode(example['src'], max_length=512, truncation=True, add_special_tokens=False))
        }

    def my_load(self, splits):
        return [self._load(name) for name in splits]

    @staticmethod
    def convert_to_features(example_batch, tokenizer):
        q1 = tokenizer.batch_encode_plus(example_batch['src'], max_length=128, truncation=True, add_special_tokens=False,)#padding='max_length')
        lengths = [len(sublist) for sublist in q1['input_ids']]
        if 'src_graph' in example_batch:
            src_atom = []
            src_edge = []
            src_charge = []
            for i in example_batch['src_graph']:
                mol =  Chem.MolFromSmiles(i)
                smiles = ''
                src_edge.append([])
                src_charge.append([])
                for atom in mol.GetAtoms():
                    atom_name = atom.GetSymbol()
                    charge = atom.GetFormalCharge()
                    if 'sep' in example_batch and atom_name + charge_num2str[charge] == 'H-':
                        smiles += '<gsep>'
                    else:
                        smiles += '<' + atom_name + charge_num2str[charge]  +'>'
                    src_charge[-1].append(charge)

                for bond in mol.GetBonds():
                    begin_atom_idx = bond.GetBeginAtomIdx()
                    end_atom_idx = bond.GetEndAtomIdx()
                    bond_type = bond.GetBondTypeAsDouble()
                    if bond_type == 1.5:
                        bond_type = 4
                    src_edge[-1].append([begin_atom_idx,end_atom_idx,int(bond_type)])

                src_atom.append('<gstart>' + smiles + '<gend>')
            q1_graph = tokenizer.batch_encode_plus(src_atom, max_length=128, truncation=True, add_special_tokens=False,)#padding='max_length')
            lengths1 = [len(sublist) for sublist in q1_graph['input_ids']]

        trg_atom = []
        trg_edge = []
        trg_charge = []
        for i in example_batch['trg']:
            mol =  Chem.MolFromSmiles(i)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            smiles = ''
            trg_edge.append([])
            trg_charge.append([])
            for atom in mol.GetAtoms():
                atom_name = atom.GetSymbol()
                charge = atom.GetFormalCharge()
                smiles += '<' + atom_name + charge_num2str[charge]  +'>'
                trg_charge[-1].append(charge)

            for bond in mol.GetBonds():
                begin_atom_idx = bond.GetBeginAtomIdx()
                end_atom_idx = bond.GetEndAtomIdx()
                bond_type = bond.GetBondTypeAsDouble()
                trg_edge[-1].append([begin_atom_idx,end_atom_idx,int(bond_type)])

            trg_atom.append('<gstart>' + smiles + '<gend>')

        q2 = tokenizer.batch_encode_plus(trg_atom, max_length=128, truncation=True, add_special_tokens=False)#,padding='max_length')
        lengths2 = [len(sublist) for sublist in q2['input_ids']]

        if 'src_graph' in example_batch:
            encodings = {
                'source': q1['input_ids'],
                'source_node': q1_graph['input_ids'],
                'source_edge': src_edge,
                'source_graph': example_batch['src_graph'],
                'target': q2['input_ids'],
                'target_edge': trg_edge,
                'target_graph': example_batch['trg'],
                'cid': example_batch['cid'],
            }
        else:
            encodings = {
                'source': q1['input_ids'],
                'target': q2['input_ids'],
                'target_edge': trg_edge,
                'target_graph': example_batch['trg'],
                'cid': example_batch['cid'],
            }

        return encodings

    @staticmethod
    def collate_fn(batch_input, tokenizer):

        text_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source']
        ) for d in batch_input], batch_first=True)

        if 'source_node' in batch_input[0]:
            source_graph_ids = pad_sequence([torch.tensor(
                d['source_node']
            ) for d in batch_input], batch_first=True)
            source_edge_input = torch.zeros(text_ids.shape[0],source_graph_ids.size(1) - 2,source_graph_ids.size(1) - 2).to(torch.int)

        target_graph_ids = pad_sequence([torch.tensor(
            d['target']  + [tokenizer.sep_token_id]
        ) for d in batch_input], batch_first=True)
        target_edge_input = torch.zeros(text_ids.shape[0],target_graph_ids.size(1) - 3,target_graph_ids.size(1) - 3).to(torch.int)

        input_ids = torch.cat((text_ids,source_graph_ids, target_graph_ids),dim = -1)\
                if 'source_node' in batch_input[0] else torch.cat((text_ids,target_graph_ids),dim = -1)
        attention_mask = torch.ones_like(input_ids)
        source_size = text_ids.size(1) + source_graph_ids.size(1) if 'source_node' in batch_input[0] else text_ids.size(1)
        target_mask = torch.stack([torch.cat([
            torch.zeros(source_size + 1), torch.ones(input_ids.size(1) - source_size - 1)
        ]) for d in batch_input])

        assert input_ids.size() == attention_mask.size() == target_mask.size()

        for i,d in enumerate(batch_input):
            if 'source_node' in batch_input[0]:
                for (start,end,bond) in d['source_edge']:
                    if start >= 125 or end >= 125:
                        continue
                    source_edge_input[i,start,end] = bond
                    source_edge_input[i,end,start] = bond
            for (start,end,bond) in d['target_edge']:
                if start >= 125 or end >= 125:
                    continue
                target_edge_input[i,start,end] = bond
                target_edge_input[i,end,start] = bond

        if 'source_node' in batch_input[0]:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_mask': target_mask,
                'source_edge_input' : source_edge_input,
                'source_start': text_ids.size(1) + 1,
                'target_edge_input' : target_edge_input,
                'target_start': text_ids.size(1) + source_graph_ids.size(1) + 1
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_mask': target_mask,
                'target_edge_input' : target_edge_input,
                'target_start': text_ids.size(1) + 1
            }

class QQPLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QQPLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'qqp'

    @staticmethod
    def add_prompt(example):
        #example['src'] = '"' + example['src'] + '" is equal to "'
        example['src'] = "Write the similar sentence: "+ example['src']
        example['trg'] = example['trg']
        return example

class CHEBILoader(GraphLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(CHEBILoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'CHEBI'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" is the description of molecular:"'
        #example['src'] = "Write the similar sentence: "+ example['src']
        example['trg'] = example['trg']
        return example

class QTLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(QTLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'Q-T'

    @staticmethod
    def add_prompt(example):
        example['src'] = ' Answer: ' + example['src'] + ' Question: '
        example['trg'] = example['trg']
        return example


class WikiLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(WikiLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'wiki_alignment'

    @staticmethod
    def add_prompt(example):
        example['src'] = '"' + example['src'] + '" can be summarized as: '
        example['trg'] = example['trg']
        return example

class CCLoader(ConditionalLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(CCLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'CC'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src'] + ' - '
        example['trg'] = example['trg']
        return example


class DiffusionLoaderWithElectra(DiffusionLoader):
    def __init__(self, model_tokenizer, electra_tokenizer, electra_model):
        super().__init__(model_tokenizer)
        self.electra_tokenizer = electra_tokenizer
        self.electra_model = electra_model

    def _load(self, task_name, split):
        dataset = datasets.load_dataset(f'./dataloaders/{task_name}.py', split=split)
        print(f'Example in {split} set:')
        print(dataset[0])
        dataset = dataset.map(partial(self.new_convert_to_features, model_tokenizer=self.tokenizer, electra_tokenizer=self.electra_tokenizer, electra_model=self.electra_model), batched=True, remove_columns='text')
        return dataset

    @staticmethod
    def new_convert_to_features(example_batch, model_tokenizer, electra_tokenizer, electra_model):
        input_encodings = model_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, add_special_tokens=False)
        electra_encodings = electra_tokenizer.batch_encode_plus(example_batch['text'], max_length=256, truncation=True, padding=True, return_tensors='pt', add_special_tokens=False)
        for k in electra_encodings.keys():
            electra_encodings[k] = electra_encodings[k].cuda()
        position = electra_encodings['attention_mask'].count_nonzero(1)
        with torch.no_grad():
            logits = electra_model(**electra_encodings)


        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'electra_logits': [logits[i][:position[i]] for i in range(position.size(0))]
        }

        return encodings

class MOIGENLoader(GraphLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(MOIGENLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'MOIGEN'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src']
        example['trg'] = example['trg']
        return example

class MOIretroLoader(GraphLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(MOIretroLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'MOIretro'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src']
        example['trg'] = example['trg']
        return example

class MOIfpLoader(GraphLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(MOIfpLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'MOIfp'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src']
        example['trg'] = example['trg']
        return example

class MOIrpLoader(GraphLoader):
    def __init__(self, tokenizer, return_source_length=False):
        super(MOIrpLoader, self).__init__(tokenizer, return_source_length)
        self.task_name = 'MOIrp'

    @staticmethod
    def add_prompt(example):
        example['src'] = example['src']
        example['trg'] = example['trg']
        return example