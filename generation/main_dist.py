import functools
import os
import sys
import random
import numpy as np
import argparse
import torch
from dataloader import QQPLoader, QTLoader, CHEBILoader, charge_num2str
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
    parser.add_argument("--epochs", default=100, type=int, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False)
    parser.add_argument("--task_name", default='CHEBI', type=str, required=False)
    parser.add_argument("--lr", default=5e-5, type=float, required=False)
    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--word_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=2000, type=int, required=False)
    parser.add_argument("--eval_step_size", default=40, type=int, required=False)
    parser.add_argument("--accumulation_steps", default=16, type=int, required=False)
    parser.add_argument("--hybrid_lambda", default=3e-4, type=float, required=False)
    parser.add_argument("--eval_steps", default=20000, type=int, required=False)
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
    parser.add_argument("--save_path", default='./new_save/42.dist_pretrain/', type=str, required=False)
    parser.add_argument("--pretrain", default=False, type=bool, required=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))

    set_seed(args)

    Dataloaders = {
        'qqp': QQPLoader,
        'QT': QTLoader,
        'CHEBI': CHEBILoader,
        #'pubchem': pubchemLoader,
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

    tokenizer = tok_cls.from_pretrained(args.model_name_or_path,local_files_only=True)
    print(len(tokenizer))

    ast = {'graph_start_token': '<gstart>', 'graph_end_token': '<gend>'}

    tokenizer.add_tokens(['<gstart>', '<gend>'])
    elements = periodictable.elements
    add_lst = []
    for element in elements:
        for i in range(-4,5):
            add_lst.append('<' + element.symbol + charge_num2str[i] + '>')

    tokenizer.add_tokens(add_lst)
    print(len(tokenizer))

    word_freq = torch.zeros(len(tokenizer))
    assert word_freq.size(0) == len(tokenizer)

    def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf


    word_freq = word_freq_preprocess_fn(word_freq)

    word_freq_edge = torch.zeros(6)
    word_freq_edge = word_freq_preprocess_fn(word_freq_edge)

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError
    print(args.sample_strategy)
    diffusion_schedule = diffusion_condition.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_condition.MaskDiffusion(
        dim=len(tokenizer),
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        word_freq=word_freq,
        word_freq_edge=word_freq_edge,
        word_freq_lambda=args.word_freq_lambda,
        device=device,
        edge_dim = 6,
    )

    if args.load_step > 0:
        ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
    cfg = cfg_cls.from_pretrained(args.model_name_or_path,local_files_only=True)
    cfg.overall_timestep = diffusion_instance.num_steps
    cfg.num_edges_type = 5
    cfg.seperate = True
    cfg.graph_token_num = len(tokenizer) - cfg.vocab_size

    if args.from_scratch:
        model = model_cls(cfg).to(device)
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=cfg, local_files_only=True).to(device)
    else:
        model = model_cls(cfg).to(device)
        model.load_state_dict(ckpt['model'])

    model.resize_token_embeddings(len(tokenizer))

    #model.load_state_dict(torch.load('./new_save/40.dist_ori/best(479999).th', map_location=lambda storage, loc: storage.cuda(local_rank))['model'])
    '''
    ckpt = torch.load('./new_save/41.dist_pubchem/best(899999).th',map_location=device)['model']
    new_checkpoint = {} 
    for k,value in ckpt.items():
        key = k.split('module.')[-1]
        new_checkpoint[key] = value
    model.load_state_dict(new_checkpoint, strict=True)
    '''
    i=-1

    model = DDP(model)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=10000)

    train_data, dev_data = Loader(tokenizer=tokenizer).my_load(splits=['train', 'validation'])


    if dist.get_rank() == 0:
        logger = set_logger(args)
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[3])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                               num_workers=1, pin_memory=True, sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size * 2, collate_fn=functools.partial(Loader.collate_fn, tokenizer=tokenizer),
                                             num_workers=4, pin_memory=True, sampler=dev_sampler)


    if args.load_step > 0:
        optimizer.load_state_dict(ckpt['optimizer'])
        warmup_scheduler.load_state_dict(ckpt['warmup_scheduler'])
    model.train()

    def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask, edge_input, target_start, position_ids):
        new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
        output = model(input_ids=new_input_ids, edge_input = edge_input, attention_mask=attention_mask, target_start = target_start, position_ids=position_ids)
        return output['logits'], output['edge_logits']

    if dist.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        best_dev_elbo = float('inf')


    train_loss = .0
    hybrid_loss_token = .0
    base_loss_token = .0
    hybrid_loss_edge = .0
    base_loss_edge = .0
    nan_count = 0
    loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]

    for epoch in range(args.epochs):
        if epoch == 30 or epoch == 60 or epoch == 90:
            args.accumulation_steps *= 2
        print(args.accumulation_steps)
        train_loader.sampler.set_epoch(epoch)
        dev_loader.sampler.set_epoch(epoch)
        for batch in tqdm(train_loader):
            i += 1
            for k, v in batch.items():
                if k != 'target_start':
                    batch[k] = v.to(device)
            t = diffusion_instance.sample_t()

            position_ids_text = create_position_ids_from_input_ids(batch['input_ids'][:,:batch['target_start']-1],cfg.pad_token_id)
            position_ids_graph = create_position_ids_from_input_ids(batch['input_ids'][:,batch['target_start']-1:],cfg.pad_token_id)
            position_ids = torch.cat((position_ids_text,position_ids_graph),dim = -1)

            metrics = diffusion_condition.compute_kl_reverse_process(
                batch['input_ids'],
                t.to(device),
                denoise_fn=functools.partial(
                    denoise_fn,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_mask=batch['target_mask'],
                    target_start = batch['target_start'],
                    position_ids = position_ids
                ),
                edge = batch['target_edge_input'],
                diffusion=diffusion_instance,
                target_mask=batch['target_mask'],
                hybrid_lambda=args.hybrid_lambda,
                predict_x0=args.predict_x0,
                word_freq_logits=torch.zeros_like(batch['input_ids']),
                target_start = batch['target_start'],
            )

            loss = metrics['loss']
            dist.all_gather(loss_list, loss)

            if torch.stack(loss_list).isnan().any():
                nan_count += 1
                if dist.get_rank() == 0:
                    logger.warning(f'NaN encountered {nan_count} times')
                continue
            train_loss += loss.item()
            hybrid_loss_token += metrics['hybrid_loss_token'].item()
            base_loss_token += metrics['base_loss_token'].item()
            hybrid_loss_edge += metrics['hybrid_loss_edge'].item()
            base_loss_edge += metrics['base_loss_edge'] .item()
            loss = loss / args.accumulation_steps
            loss.backward()
            # diffusion_instance.update_loss(t.numpy(), loss.item())
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            if i % args.accumulation_steps == args.accumulation_steps - 1:
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                warmup_scheduler.step()

            if dist.get_rank() == 0:
                if i % args.logging_steps == args.logging_steps - 1:
                    logger.info(f'Loss at step {i} is {train_loss / args.logging_steps}')
                    logger.info(f'hybrid_loss_token at step {i} is {hybrid_loss_token / args.logging_steps}')
                    logger.info(f'base_loss_token at step {i} is {base_loss_token / args.logging_steps}')
                    logger.info(f'hybrid_loss_edge at step {i} is {hybrid_loss_edge / args.logging_steps}')
                    logger.info(f'base_loss_edge at step {i} is {base_loss_edge / args.logging_steps}')
                    logger.info('\n')
                    train_loss = .0
                    hybrid_loss_token = .0
                    base_loss_token = .0
                    hybrid_loss_edge = .0
                    base_loss_edge = .0
            
            if i % args.eval_steps == args.eval_steps - 1:
                nan_count_in_dev = 0
                model.eval()
                dev_metrics = {
                    'elbo': .0,
                    'elbo_in_bits_per_dim': .0,
                    # 'likelihood': .0,
                    # 'prior': .0,
                }
                with torch.no_grad():
                    '''
                    for dev_batch in tqdm(dev_loader):
                        for k, v in dev_batch.items():
                            dev_batch[k] = v.to(device)
                        batch_dev_metrics = diffusion_condition.discrete_diffusion_elbo(
                            dev_batch['input_ids'],
                            denoise_fn=functools.partial(
                                denoise_fn,
                                input_ids=dev_batch['input_ids'],
                                attention_mask=dev_batch['attention_mask'],
                                target_mask=dev_batch['target_mask']
                            ),
                            diffusion=diffusion_instance,
                            target_mask=dev_batch['target_mask'],
                            normalize_without_padding=True,
                            eval_step_size=args.eval_step_size,
                            word_freq_logits=torch.zeros_like(dev_batch['input_ids'])
                        )

                        m = torch.tensor(0., device=device)
                        for name in dev_metrics.keys():
                            m = batch_dev_metrics[name].squeeze()
                            temp = m
                            if not torch.isnan(temp):
                                dev_metrics[name] += temp
                            else:
                                nan_count_in_dev += 1
                                logger.warning(f'NaN encountered {nan_count_in_dev} times in dev')

                    for name in dev_metrics.keys():
                        dev_metrics[name] /= (len(dev_data) - nan_count_in_dev * 2 * args.batch_size)
                        logging.info(f"eval {dev_metrics[name]}: name = {name}")
                        #fitlog.add_metric(dev_metrics[name], name=name, step=i)
                    if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                        best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                        logging.info(f"best {dev_metrics['elbo_in_bits_per_dim']}: name = elbo_in_bits_per_dim")
                        #fitlog.add_best_metric(dev_metrics['elbo_in_bits_per_dim'], name='dev_elbo_in_bits_per_dim')
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'warmup_scheduler': warmup_scheduler.state_dict(),
                        }, f'./{save_path}/best({i}).th')
                    '''
                    if dist.get_rank() == 0:
                        torch.save({
                            'model': model.state_dict(),
                        }, f'./new_save/42.dist_pretrain/best({i}).th')
                model.train()
            
            #if i % args.save_steps == args.save_steps - 1:
            #    torch.save({
            #        'model': model.state_dict(),
            #        'optimizer': optimizer.state_dict(),
            #        'warmup_scheduler': warmup_scheduler.state_dict(),
            #    }, f'{save_path}/{i}.th')
