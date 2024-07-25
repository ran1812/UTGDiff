import torch
import os
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from transformers import BertTokenizer as ElasticBertTokenizer
from models.modeling_roberta_new import RobertaForMaskedLM
from sample import Categorical, WholeWordMasking
import time
from tqdm import tqdm
from compute_metric import self_bleu, dist1, div4
import nltk
import argparse
from models.modeling_bert import BertForMaskedLM
from dataloader import QQPLoader, QTLoader, CHEBILoader, charge_num2str
import diffusion_condition as diffusion
import functools
from MBR_decoding import selectBest
from torch.nn.utils.rnn import pad_sequence
import periodictable

from evaluation import graph2smile

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=15, type=int, required=False)
parser.add_argument("--step_size", default=10, type=int, required=False, help='Time step size during inference')
parser.add_argument("--task_name", default='CHEBI', type=str, required=False)
parser.add_argument("--ckpt_path", default='./new_save/28.simple_charge_lam/best(1909999).th', type=str, required=False)
parser.add_argument("--MBR_size", default=1, type=int, required=False, help=r'The MBR size \mathcal{S}. Generates that many sentences for 1 source sentence.')
parser.add_argument("--seq_len", default=256, type=int, required=False, help='Max seq length in generation')
args = parser.parse_args()

step_size = args.step_size
device = 'cuda:0'
model_name = 'roberta-base'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 2000
schedule = 'mutual'
topk = args.topk
task_name = args.task_name
model_ckpt_path = args.ckpt_path
temperature = 1.0
batch_size = 2
MBR_size = args.MBR_size

if not os.path.exists('./generation_results'):
    os.mkdir('generation_results')

Dataloaders = {
    'qqp': QQPLoader,
    'QT': QTLoader,
    'CHEBI': CHEBILoader,
}

if model_name in ['roberta-base']:
    model_cls = RobertaForMaskedLM
    cfg_cls = RobertaConfig
    tok_cls = RobertaTokenizer
elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
    model_cls = BertForMaskedLM
    cfg_cls = BertConfig
    tok_cls = BertTokenizer
else:
    raise NotImplementedError

tokenizer = tok_cls.from_pretrained(model_name, local_files_only=True)

ast = {'graph_start_token': '<gstart>', 'graph_end_token': '<gend>'}

tokenizer.add_tokens(['<gstart>', '<gend>'])
elements = periodictable.elements
add_lst = []
for element in elements:
    for i in range(-4,5):
        add_lst.append('<' + element.symbol + charge_num2str[i] + '>')
tokenizer.add_tokens(add_lst)

if sample_strategy == 'Categorical':
    sample_cls = Categorical()
elif sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer)
else:
    raise ValueError


word_freq = torch.ones(len(tokenizer))
def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf

word_freq = word_freq_preprocess_fn(word_freq)

word_freq_edge = torch.ones(6)
word_freq_edge = word_freq_preprocess_fn(word_freq_edge)

diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
diffusion_instance = diffusion.MaskDiffusion(
    dim=len(tokenizer),
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq=word_freq,
    device=device,
    )

cfg = cfg_cls.from_pretrained(model_name, local_files_only=True)
cfg.overall_timestep = diffusion_instance.num_steps
cfg.num_edges_type = 5
cfg.seperate = True

model = model_cls(cfg).to(device)
model.resize_token_embeddings(len(tokenizer))
ckpt = torch.load(model_ckpt_path,map_location=device)

# print(ckpt['model'])

model.load_state_dict(ckpt['model'])

def denoise_fn(input_ids, corrupted_input_ids, timestep, attention_mask, target_mask, edge_input, target_start):
    new_input_ids = torch.where(target_mask.bool(), corrupted_input_ids, input_ids)
    output = model(input_ids=new_input_ids, edge_input = edge_input, attention_mask=attention_mask, target_start = target_start)
    return output['logits'], output['edge_logits']

model.eval()


def process_fn_in_collate(wf):
    return wf - wf.mean()


def collate_fn(batch_input):
    text_ids = pad_sequence([torch.tensor(
            [tokenizer.cls_token_id] + d['source']
        ) for d in batch_input], batch_first=True)

    graph_ids = pad_sequence([torch.tensor([tokenizer.encode('<gstart>')[1]] + [tokenizer.mask_token_id] * 127) for d in batch_input], batch_first=True)

    input_ids = torch.cat((text_ids,graph_ids),dim = -1)
    attention_mask = torch.ones_like(input_ids)

    target_mask = torch.stack([torch.cat([
        torch.zeros(text_ids.size(1) + 1), torch.ones(input_ids.size(1) - text_ids.size(1) - 1)
    ]) for d in batch_input])
    target_start0 = torch.tensor([text_ids.size(1) + 1 for d in batch_input]).long()

    edge_input = torch.zeros(input_ids.shape[0],graph_ids.size(1) - 3,graph_ids.size(1) - 3).to(torch.int)

    for i,d in enumerate(batch_input):
        for (start,end,bond) in d['edge']:
            edge_input[i,start,end] = bond
            edge_input[i,end,start] = bond

    edge_init = torch.ones_like(edge_input) * (cfg.num_edges_type)
    assert input_ids.size() == attention_mask.size() == target_mask.size()

    return {
        'input_ids': input_ids.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'attention_mask': attention_mask.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'target_mask': target_mask.repeat(1, MBR_size).view(-1, input_ids.size(-1)),
        'edge_input': edge_input.repeat(1, MBR_size, 1).view(-1, edge_input.size(-1),edge_input.size(-1)),
        'edge_init': edge_init.repeat(1, MBR_size, 1).view(-1, edge_input.size(-1),edge_input.size(-1)),
        'target_start': text_ids.size(1) + 1,
        'target_start0': target_start0,
    }

test_data = Dataloaders[task_name](tokenizer=tokenizer).my_load(splits=['test'])[0]
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)

s_bleu = 0.
dist_1 = 0.
div_4 = 0.
generated_smiles = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            if k != 'target_start':
                batch[k] = v.to(device)

        outputs = diffusion.discrete_diffusion_predict_fn(
            input_ids=batch['input_ids'],
            target_mask=batch['target_mask'],
            denoise_fn=functools.partial(
                        denoise_fn,
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        target_mask=batch['target_mask'],
                        edge_input = batch['edge_input'],
                        target_start = batch['target_start']
                    ),
            diffusion=diffusion_instance,
            predict_x0=predict_x0,
            sample_cls=sample_cls,
            step_size=step_size,
            topk=topk,
            show_process=False,
            temperature=temperature,
            MBR_size=MBR_size
        )
        state = outputs['final_state']#.view(batch_size, MBR_size, -1)
        edge = batch['edge_input']#.view(batch_size, MBR_size, -1)
        generated_smiles = graph2smile(state,edge,tokenizer)
        with open(f'./generation_results/{task_name}_{args.topk}_MBR_(25_166_1).txt', 'a+') as f_mbr:
            for num,i in enumerate(generated_smiles):
                f_mbr.write(str(i)+' ')
                if num % 3 ==2:
                    f_mbr.write('\n')
