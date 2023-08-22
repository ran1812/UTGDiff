import torch
import torch.nn as nn
import torch_geometric
import argparse
from data import BasicData,get_dataloader,get_graph_data

parser = argparse.ArgumentParser(description='main')
parser.add_argument('--dataset', type=str,default='CHEBI',choices=['PCDes','CHEBI'],
                    help='directory where data is located')
parser.add_argument('--text_trunc_length', type=int, nargs='?', default=256,
                    help='Text truncation length.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Size of data batch.')
parser.add_argument('--epochs', type=int, default=1,
                    help='epochs')

args = parser.parse_args()  
data_path = './dataset/' + args.dataset

if args.dataset == 'CHEBI':
    path_train = data_path + '/train.txt'
    path_val = data_path + '/val.txt'
    path_test = data_path + '/test.txt'
    graph_data_path = data_path + '/mol_graphs.zip'
    path_molecules = data_path + '/ChEBI_defintions_substructure_corpus.cp'
    path_token_embs = data_path + './token_embedding_dict.npy'

if args.dataset == 'PCDes':
    path_train = data_path + '/train_des.txt'
    path_val = data_path + '/val_des.txt'
    path_test = data_path + '/test_des.txt'
    graph_data_path = data_path + '/align_smiles.txt'
    path_molecules = None
    path_token_embs = None

gd = BasicData(args.text_trunc_length, path_train, path_val, path_test, path_molecules, path_token_embs,args.dataset)

params = {'batch_size': args.batch_size}

training_generator, validation_generator, test_generator = get_dataloader(gd, params)
graph_batcher_train, graph_batcher_val, graph_batcher_test = get_graph_data(gd, graph_data_path,args.dataset)

for epochs in range(args.epochs):
    for i, d in enumerate(validation_generator):
        batch = d
        graph_batch = graph_batcher_val(batch['molecule']['cid'])
        text_mask = batch['text']['attention_mask'].bool()

        text = batch['text']['input_ids']