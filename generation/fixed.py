import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
from dataloader import QQPLoader, QTLoader, CHEBILoader, charge_num2str, fixedCHEBILoader
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert_new import BertForMaskedLM
from models.modeling_roberta_new import RobertaForMaskedLM
import diffusion_condition_new as diffusion_condition
from torch.optim import AdamW
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime

import logging
from evaluation import graph2smile, graph2smile_new, graph2smile_newpos
import periodictable
from models.modeling_roberta_new import create_position_ids_from_input_ids

def set_logger(args):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(args.save_path + 'train_log.log')
    file_handler.setLevel(logging.DEBUG)

    # 创建一个流处理器，用于将日志输出到终端
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=600, type=int, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False)
    parser.add_argument("--task_name", default='fixed_CHEBI', type=str, required=False)
    parser.add_argument("--lr", default=5e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=2000, type=int, required=False)
    parser.add_argument("--eval_step_size", default=40, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=4, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--eval_steps", default=10000, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=200, type=int, required=False)
    parser.add_argument("--save_steps", default=2000, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    parser.add_argument("--save_path", default='./new_save/38.addcorrupt2/', type=str, required=False)
    parser.add_argument("--pretrain", default=False, type=bool, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    local_rank = 3
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    set_seed(args)

    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
        'CHEBI': CHEBILoader,
        'fixed_CHEBI': fixedCHEBILoader
    }

    Loader = Dataloaders[args.task_name]

    save_path = f'./new_save'
    
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased','allenai/scibert_scivocab_uncased']:
        model_cls = BertForMaskedLM
        cfg_cls = BertConfig
        tok_cls = BertTokenizer
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
        cfg_cls = RobertaConfig
        tok_cls = RobertaTokenizer
    else:
        raise NotImplementedError
    '''
    tokenizer = tok_cls.from_pretrained('./pretrain_mixed4',local_files_only=True)
    cfg = cfg_cls.from_pretrained('./pretrain_mixed4',local_files_only=True)
    model = model_cls.from_pretrained('./pretrain_mixed4', config=cfg, local_files_only=True).to(device)
    '''
    tokenizer = tok_cls.from_pretrained(args.model_name_or_path,local_files_only=True)
    ast = {'graph_start_token': '<gstart>', 'graph_end_token': '<gend>'}
    tokenizer.add_tokens(['<gstart>', '<gend>'])
    elements = periodictable.elements
    add_lst = []
    for element in elements:
        for i in range(-4,5):
            add_lst.append('<' + element.symbol + charge_num2str[i] + '>')
    tokenizer.add_tokens(add_lst)

    cfg = cfg_cls.from_pretrained(args.model_name_or_path,local_files_only=True)
    cfg.num_edges_type = 5
    cfg.seperate = True
    cfg.graph_token_num = len(tokenizer) - cfg.vocab_size

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg, local_files_only=True).to(device)

    model.resize_token_embeddings(len(tokenizer))
    
    ckpt = torch.load('./new_save/43.dist_pretrain1/best(959999).th',map_location=device)['model']
    new_checkpoint = {} 
    for k,value in ckpt.items():
        key = k.split('module.')[-1]
        new_checkpoint[key] = value
    model.load_state_dict(new_checkpoint, strict=True)
    
    
    i=-1
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

    test_data = Loader(tokenizer=tokenizer).my_load(splits=['test'])[0]

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                               num_workers=1, pin_memory=True)

    model.eval()

    for epoch in range(args.epochs):
        for batch in tqdm(test_loader):
            i += 1
            for k, v in batch.items():
                if k != 'target_start':
                    batch[k] = v.to(device)
            position_ids_text = create_position_ids_from_input_ids(batch['input_ids'][:,:batch['target_start']-1],cfg.pad_token_id)
            position_ids_graph = create_position_ids_from_input_ids(batch['input_ids'][:,batch['target_start']-1:],cfg.pad_token_id)
            position_ids = torch.cat((position_ids_text,position_ids_graph),dim = -1)
            output = model(input_ids=batch['input_ids'], edge_input = batch['target_edge_input'], attention_mask=batch['attention_mask']
                           , target_start = batch['target_start'], position_ids=position_ids)
            logits = output['logits']
            edge_logits = output['edge_logits']
            probs = torch.nn.Softmax(dim=-1)(logits)
            edge_probs = torch.nn.Softmax(dim=-1)(edge_logits)
            print(probs.argmax(-1)[0])
            print(batch['input_ids'][0])
            print(edge_probs.argmax(-1)[0])
            print(edge_probs[0][0][1])
            print(torch.where(batch['target_edge_input'][0]==1),torch.where(batch['target_edge_input'][0]==2),torch.where(batch['target_edge_input'][0]==3),torch.where(batch['target_edge_input'][0]==4))
            print(torch.where(edge_probs.argmax(-1)[0]==1),torch.where(edge_probs.argmax(-1)[0]==2),torch.where(edge_probs.argmax(-1)[0]==3),torch.where(edge_probs.argmax(-1)[0]==4))

            generated_smiles = graph2smile_newpos(probs.argmax(-1),edge_probs.argmax(-1), position_ids, tokenizer)
            print(generated_smiles)
            with open(f'./fixed.txt', 'a+') as f_mbr:
                for num,i in enumerate(generated_smiles):
                    f_mbr.write(str(i)+' ')
                    f_mbr.write('\n')
            exit()

            