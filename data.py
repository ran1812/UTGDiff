import numpy as np
import csv
import shutil
import torch
import os
import os.path as osp
import zipfile
from torch.utils.data import Dataset, DataLoader
import networkx as nx

from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.data import Data, Batch

from transformers import BertTokenizerFast

from rdkit import Chem
from rdkit.Chem import Draw


class BasicData():
  def __init__(self, text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs,name):
    self.path_train = path_train
    self.path_val = path_val
    self.path_test = path_test
    self.path_molecules = path_molecules
    self.path_token_embs = path_token_embs
    self.name = name

    self.text_trunc_length = text_trunc_length 

    self.text_tokenizer = BertTokenizerFast.from_pretrained("allenai/scibert_scivocab_uncased")

    if path_molecules is not None:  
      self.load_substructures()

    self.store_descriptions()
    
  def load_substructures(self):
    self.molecule_sentences = {}
    self.molecule_tokens = {}

    total_tokens = set()
    self.max_mol_length = 0
    with open(self.path_molecules) as f:
      for line in f:
        spl = line.split(":")
        cid = spl[0]
        tokens = spl[1].strip()
        self.molecule_sentences[cid] = tokens
        t = tokens.split()
        total_tokens.update(t)
        size = len(t)
        if size > self.max_mol_length: self.max_mol_length = size


    self.token_embs = np.load(self.path_token_embs, allow_pickle = True)[()]

  def store_descriptions(self):
    self.descriptions = {}
    
    self.mols = {}

    self.training_cids = []
    self.validation_cids = []
    self.test_cids = []
    #get training set cids...
    read_lst = [self.path_train, self.path_val, self.path_test]
    cids = [self.training_cids,self.validation_cids,self.test_cids]
    tot = 0
    for i in range(3):
      path = read_lst[i]
      cid = cids[i]
      with open(path,encoding='utf-8') as f:
        if self.name == 'CHEBI':
          reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, fieldnames = ['cid', 'mol2vec', 'desc'])
          for n, line in enumerate(reader):
            self.descriptions[line['cid']] = line['desc']
            self.mols[line['cid']] = line['mol2vec']
            cid.append(line['cid'])
        elif self.name == 'PCDes':
          context = f.readlines()
          for line in context:
            self.descriptions[str(tot)] = line
            cid.append(str(tot))
            tot+=1
    print(len(self.training_cids))
    print(len(self.validation_cids))
    print(len(self.test_cids))

  def generate_examples_train(self):
    """Yields examples."""

    np.random.shuffle(self.training_cids)

    for cid in self.training_cids:
      text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, max_length=self.text_trunc_length,
                                        padding='max_length', return_tensors = 'np')

      yield {
          'cid': cid,
          'input': {
              'text': {
                'input_ids': text_input['input_ids'].squeeze(),
                'attention_mask': text_input['attention_mask'].squeeze(),
              },
              'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " ") if self.name == 'CHEBI' else torch.zeros(1),
                    'cid' : cid
              },
          },
      }


  def generate_examples_val(self):
    """Yields examples."""

    np.random.shuffle(self.validation_cids)

    for cid in self.validation_cids:
        text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, padding = 'max_length', 
                                         max_length=self.text_trunc_length, return_tensors = 'np')

        mol_input = []

        yield {
            'cid': cid,
            'input': {
                'text': {
                  'input_ids': text_input['input_ids'].squeeze(),
                  'attention_mask': text_input['attention_mask'].squeeze(),
                },
                'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " ")  if self.name == 'CHEBI' else torch.zeros(1),
                    'cid' : cid
                }
            },
        }


  def generate_examples_test(self):
    """Yields examples."""

    np.random.shuffle(self.test_cids)

    for cid in self.test_cids:
        text_input = self.text_tokenizer(self.descriptions[cid], truncation=True, padding = 'max_length', 
                                         max_length=self.text_trunc_length, return_tensors = 'np')

        mol_input = []

        yield {
            'cid': cid,
            'input': {
                'text': {
                  'input_ids': text_input['input_ids'].squeeze(),
                  'attention_mask': text_input['attention_mask'].squeeze(),
                },
                'molecule' : {
                    'mol2vec' : np.fromstring(self.mols[cid], sep = " ")  if self.name == 'CHEBI' else torch.zeros(1),
                    'cid' : cid
                }
            },
        }

class MolDataset(Dataset):
  def __init__(self, gen, length):
      self.gen = gen
      self.it = iter(self.gen())

      self.length = length

  def __len__(self):
      return self.length


  def __getitem__(self, index):
      try:
        ex = next(self.it)
      except StopIteration:
        self.it = iter(self.gen())
        ex = next(self.it)

      X = ex['input']

      return X

def get_dataloader(data_generator, params):

    training_set = MolDataset(data_generator.generate_examples_train, len(data_generator.training_cids))
    validation_set = MolDataset(data_generator.generate_examples_val, len(data_generator.validation_cids))
    test_set = MolDataset(data_generator.generate_examples_test, len(data_generator.test_cids))

    training_generator = DataLoader(training_set, **params)
    validation_generator = DataLoader(validation_set, **params)
    test_generator = DataLoader(test_set, **params)


    return training_generator, validation_generator, test_generator

class MoleculeGraphDataset(GeoDataset):
    def __init__(self, root, cids, data_path, gt, name,transform=None, pre_transform=None):
        self.cids = cids
        self.data_path = data_path
        self.gt = gt
        self.name = name
        super(MoleculeGraphDataset, self).__init__(root, transform, pre_transform)
      
        self.idx_to_cid = {}
        i = 0
        for raw_path in self.raw_paths:
            if len(raw_path.split('/')) != 1:
                cid = int(raw_path.split('/')[-1][:-6])
            else:
                cid = int(raw_path.split('\\')[-1][:-6])
            self.idx_to_cid[i] = cid
            i += 1

    @property
    def raw_file_names(self):
        return [cid + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ['data_{}.pt'.format(cid) for cid in self.cids]

    def download(self):
        # Download to `self.raw_dir`.
        if self.name == 'CHEBI':
          print(self.raw_dir)
          print(osp.join(self.raw_dir, "mol_graphs.zip"))
          print(osp.exists(osp.join(self.raw_dir, "mol_graphs.zip")))
          if not osp.exists(osp.join(self.raw_dir, "mol_graphs.zip")):
              shutil.copy(self.data_path, os.path.join(self.raw_dir, "mol_graphs.zip"))
        
    def process_graph(self, raw_path):
      edge_index  = []
      x = []
      with open(raw_path, 'r') as f:
        next(f)
        for line in f: #edges
          if line != "\n":
            edge = *map(int, line.split()), 
            edge_index.append(edge)
          else:
            break
        next(f)
        for line in f: #get mol2vec features:
          substruct_id = line.strip().split()[-1]
          if substruct_id in self.gt.token_embs:
            x.append(self.gt.token_embs[substruct_id])
          else:
            x.append(self.gt.token_embs['UNK'])

        return torch.LongTensor(edge_index).T, torch.FloatTensor(x)

    def process(self):
        if self.name =='CHEBI':
          with zipfile.ZipFile(osp.join(self.raw_dir, "mol_graphs.zip"), 'r') as zip_ref:
              zip_ref.extractall(self.raw_dir)
          length = len(self.raw_paths)
        elif self.name == 'PCDes':
          with open(self.data_path,encoding='utf-8') as f:
             smiles = f.readlines()
          length = len(smiles)

        i = 0
        for idx in range(length):
            # Read data from `raw_path`.
            if i%100==0:
               print(i)
            if self.name == 'CHEBI':
              raw_path = self.raw_paths[idx]
              if len(raw_path.split('/')) != 1:
                  cid = int(raw_path.split('/')[-1][:-6])
              else:
                  cid = int(raw_path.split('\\')[-1][:-6])

              edge_index, x = self.process_graph(raw_path)
            elif self.name == 'PCDes':
              smile = smiles[idx]
              cid = idx
              edge_index, x = self.smiles_to_edge_index(smile)
            data = Data(x=x, edge_index = edge_index)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(self.idx_to_cid[idx])))
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(cid)))
        return data
    
    def smiles_to_edge_index(self,smiles):
      molecule = Chem.MolFromSmiles(smiles)
      num_atoms = molecule.GetNumAtoms()
      edge_index = []

      for bond in molecule.GetBonds():
          start_idx = bond.GetBeginAtomIdx()
          end_idx = bond.GetEndAtomIdx()
          edge_index.append((start_idx,end_idx))

      return torch.LongTensor(edge_index).T, torch.FloatTensor(torch.zeros(num_atoms,16))

#To get specific lists...

class CustomGraphCollater(object):
    def __init__(self, dataset, follow_batch = [], exclude_keys = []):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.dataset = dataset

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch) 
            
        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, cids):
        return self.collate([self.dataset.get_cid(int(cid)) for cid in cids])


def get_graph_data(data_generator, graph_data_path,name):
    root = osp.join(graph_data_path[:-len(osp.basename(graph_data_path))], 'graph-data/')
    if not os.path.exists(root):
        os.mkdir(root)
    mg_data_tr = MoleculeGraphDataset(root, data_generator.training_cids, graph_data_path, data_generator,name)
    graph_batcher_tr = CustomGraphCollater(mg_data_tr)

    mg_data_val = MoleculeGraphDataset(root, data_generator.validation_cids, graph_data_path, data_generator,name)
    graph_batcher_val = CustomGraphCollater(mg_data_val)

    mg_data_test = MoleculeGraphDataset(root, data_generator.test_cids, graph_data_path, data_generator,name)
    graph_batcher_test = CustomGraphCollater(mg_data_test)

    return graph_batcher_tr, graph_batcher_val, graph_batcher_test