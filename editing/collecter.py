# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from typing import Optional, Any, Tuple

from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers.tokenization_utils_base import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
import time

from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem

charge_num2str = {-5:'-5',-4:'-4',-3:'-3',-2:'-2',-2:'-2',-1:'-',0:'',1:'+',2:'+2',3:'+3',4:'+4',5:'+5',6:'+6',7:'+7',8:'+8'}

def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

class CollatorForMOILanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf
            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, types='node') -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if types == 'edge':
            special_tokens_mask = torch.zeros_like(inputs, dtype=torch.bool)
        elif types == 'node':
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            else:
                special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices

        if types == 'node':
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        elif types == 'edge':
            inputs[indices_replaced] = 4

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        if types == 'node':
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        elif types == 'edge':
            random_words = torch.randint(5, labels.shape, dtype=torch.int)
        inputs[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def torch_call(self, examples):

        text_ids = pad_sequence([torch.tensor(
            [self.tokenizer.cls_token_id] + d['source']
        ) for d in examples], batch_first=True)
        target_graph_ids = pad_sequence([torch.tensor(
            d['target']  + [self.tokenizer.sep_token_id]
        ) for d in examples], batch_first=True)

        target_edge_input = torch.zeros(text_ids.shape[0],target_graph_ids.size(1) - 3,target_graph_ids.size(1) - 3).to(torch.int)

        input_ids = torch.cat((text_ids,target_graph_ids),dim = -1)
        attention_mask = torch.ones_like(input_ids)

        assert input_ids.size() == attention_mask.size()

        for i,d in enumerate(examples):
            for (start,end,bond) in d['target_edge']:
                target_edge_input[i,start,end] = bond
                target_edge_input[i,end,start] = bond

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'edge_input' : target_edge_input,
            'target_start': text_ids.size(1) + 1
        }

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask, types='node'
            )
            num_node = batch["edge_input"].shape[1]
            batch["edge_input"], batch["edge_labels"] = self.torch_mask_tokens(
                batch["edge_input"].reshape(batch["edge_input"].shape[0],-1), special_tokens_mask=special_tokens_mask, types='edge'
            )
            batch["edge_input"] = batch["edge_input"].reshape(batch["edge_input"].shape[0], num_node, num_node)
            batch["edge_labels"] = batch["edge_labels"].reshape(batch["edge_labels"].shape[0], num_node, num_node)
        return batch
    
class CollatorForGIMLETLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        self.transform_in_collator= kwargs.pop('transform_in_collator')
        self.rich_features = kwargs.pop('rich_features')
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf

            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def graph_collect(self, examples):
        trg_atom = []
        trg_edge = []
        trg_charge = []
        for example_data in examples:
            i = example_data['graph']
            mol =  Chem.MolFromSmiles(i)
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
                if bond_type == 1.5:
                    bond_type = 4
                trg_edge[-1].append([begin_atom_idx,end_atom_idx,int(bond_type)])

            trg_atom.append('<gstart>' + smiles + '<gend>')

        q2 = self.tokenizer.batch_encode_plus(trg_atom, max_length=128, truncation=True, add_special_tokens=False,padding='max_length')
        
        return q2['input_ids'], trg_edge

    def torch_call(self, examples):
        text_ids = pad_sequence([torch.tensor(
                [self.tokenizer.cls_token_id] + d['input_ids'][:256]
            ) for d in examples], batch_first=True)
        text_attention_mask = pad_sequence([torch.tensor(
                [1] + d['attention_mask'][:256]
            ) for d in examples], batch_first=True)
        
        graph_ids, graph_edge = self.graph_collect(examples)
        
        target_graph_ids = pad_sequence([torch.tensor(
            d  + [self.tokenizer.sep_token_id]
        ) for d in graph_ids], batch_first=True)
        target_edge_input = torch.zeros(text_ids.shape[0],target_graph_ids.size(1) - 3,target_graph_ids.size(1) - 3).to(torch.int)

        input_ids = torch.cat((text_ids,target_graph_ids),dim = -1)
        attention_mask = torch.cat((text_attention_mask, torch.ones_like(target_graph_ids)),dim = -1)

        assert input_ids.size() == attention_mask.size() 

        for i,d in enumerate(graph_edge):
            for (start,end,bond) in d:
                if start >= 126 or end >= 126:
                    continue
                target_edge_input[i,start,end] = bond
                target_edge_input[i,end,start] = bond
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'edge_input' : target_edge_input,
            'target_start': text_ids.size(1) + 1
        }

        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )

        return batch
