import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import torch

import datasets
from datasets import load_dataset, DatasetDict

import transformers
from transformers import set_seed, AutoTokenizer, AutoConfig, AutoModelForMaskedLM, DataCollatorForLanguageModeling

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from transformers import BertTokenizer, BertConfig, RobertaTokenizer, RobertaConfig
from models.modeling_bert_new import BertForMaskedLM
from models.modeling_roberta_new import RobertaForMaskedLM
import argparse

from utils import tokenize_function_gimlet
from dataloaders.gimlet_collator import CollatorForGIMLETLanguageModeling

from transformers import TrainingArguments
from dataloader import MOIGENLoader,MOIretroLoader, MOIfpLoader, MOIrpLoader, charge_num2str
import periodictable
from collecter import CollatorForMOILanguageModeling, CollatorForGIMLETLanguageModeling

from trainer import NewTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='MOIGEN', type=str, required=False)
    parser.add_argument("--train_file", default='', type=str, required=False)
    parser.add_argument("--validation_file", default='', type=str, required=False)
    parser.add_argument("--model_name_or_path", default='roberta-base', type=str, required=False)
    parser.add_argument("--preprocessing_num_workers", default=8, type=int, required=False)
    parser.add_argument("--rich_features", default=False, type=bool, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--batch_size", default=1, type=int, required=False)
    parser.add_argument("--save_steps", default=100000, type=int, required=False)
    parser.add_argument("--epochs", default=2.0, type=float, required=False)
    return parser.parse_args()

def main():
    args = parse_args()
    training_args = TrainingArguments(output_dir="result",do_train=True,do_eval=False, overwrite_output_dir=True,
                                      per_device_train_batch_size=args.batch_size, save_steps= args.save_steps, 
                                      num_train_epochs=args.epochs,logging_steps=100)
    print(training_args)
    set_seed(training_args.seed)
    print(training_args)
    if args.model_name_or_path in ['bert-base-uncased', 'bert-large-uncased','allenai/scibert_scivocab_uncased']:
        model_cls = BertForMaskedLM
    elif args.model_name_or_path in ['roberta-base']:
        model_cls = RobertaForMaskedLM
    else:
        raise NotImplementedError

    logger = logging.getLogger(__name__)
    if args.data=='haitengzhao/molecule_property_instruction':
        raw_datasets={}
        dataset_full=load_dataset("haitengzhao/molecule_property_instruction",
                     # download_mode = "force_redownload"
                     )
        if training_args.do_train:
            raw_datasets['train']=dataset_full['chembl_pretraining']
        if training_args.do_eval:
            raw_datasets['validation']=dataset_full['chembl_zero_shot']
        raw_datasets=DatasetDict(raw_datasets)
    elif args.data == 'MOIGEN':
        Loader = MOIGENLoader
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    config.num_edges_type = 4
    config.seperate = True
    if args.from_scratch:
        model = model_cls(config).cuda()
    elif args.load_step <= 0:
        model = model_cls.from_pretrained(args.model_name_or_path, config=config).cuda()
    else:
        model = model_cls(config).cuda()
        model.load_state_dict(ckpt['model'])
    
    ast = {'graph_start_token': '<gstart>', 'graph_end_token': '<gend>'}
    tokenizer.add_tokens(['<gstart>', '<gend>'])
    elements = periodictable.elements
    add_lst = []
    for element in elements:
        for i in range(-4,9):
            add_lst.append('<' + element.symbol + charge_num2str[i] + '>')
    tokenizer.add_tokens(add_lst)
    print(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))

    if args.data == 'MOIGEN':
        train_dataset, eval_dataset = MOIGENLoader(tokenizer=tokenizer).my_load(splits=['train', 'validation'])
        data_collator = CollatorForMOILanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=None,
        )

    else:
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024

        padding = False


        tokenize_function=lambda x: tokenize_function_gimlet(examples=x, tokenizer=tokenizer, text_column_name=text_column_name, padding=padding, max_seq_length=max_seq_length, rich_features=args.rich_features, transform_in_collator=True)
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset line_by_line",
        )
        
        data_collator = CollatorForGIMLETLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability=0.15,
                pad_to_multiple_of=None,
                transform_in_collator=True,
                rich_features=False
            )

        if training_args.do_train:
            if "train" not in tokenized_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = tokenized_datasets["train"]

        if training_args.do_eval:
            if "validation" not in tokenized_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = tokenized_datasets["validation"]

    training_args.remove_unused_columns=False
    trainer = NewTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()



if __name__ == "__main__":
    main()