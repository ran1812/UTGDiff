# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from typing import Optional, Any, Tuple

from transformers import (
    DataCollatorForLanguageModeling,
)
from transformers import BatchEncoding

from collections.abc import Mapping, Sequence

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch
import numpy as np
import time

from torch.nn.utils.rnn import pad_sequence

from rdkit import Chem
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Dict, List
from transformers.models.bart.modeling_bart import shift_tokens_right

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
        self.num_edges_type = kwargs.pop('num_edges_type')
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf
            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, types='node', target_start = None) -> Tuple[Any, Any]:
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
                special_tokens_mask = [i[:target_start] + [0]*(len(i)-target_start) for i in special_tokens_mask]
                special_tokens_mask = [[0]*(len(i)) for i in special_tokens_mask]
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
            inputs[indices_replaced] = self.num_edges_type

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
        attention_mask_text = torch.ones_like(text_ids)
        attention_mask_text[text_ids == 0] = 0
        attention_mask_graph = torch.ones_like(target_graph_ids)
        attention_mask = torch.cat((attention_mask_text,attention_mask_graph),dim = -1)

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
                batch["input_ids"], special_tokens_mask=special_tokens_mask, types='node', target_start = text_ids.size(1) + 1,
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
        self.num_edges_type = kwargs.pop('num_edges_type')
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
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, types='node', target_start = None) -> Tuple[Any, Any]:
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
                special_tokens_mask = [i[:target_start] + [0]*(len(i)-target_start) for i in special_tokens_mask]
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
            inputs[indices_replaced] = self.num_edges_type

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
                [self.tokenizer.cls_token_id] + d['input_ids'][:128]
            ) for d in examples], batch_first=True)
        attention_mask_text = pad_sequence([torch.tensor(
                [1] + d['attention_mask'][:128]
            ) for d in examples], batch_first=True)

        graph_ids, graph_edge = self.graph_collect(examples)
        
        target_graph_ids = pad_sequence([torch.tensor(
            d  + [self.tokenizer.sep_token_id]
        ) for d in graph_ids], batch_first=True)
        target_edge_input = torch.zeros(text_ids.shape[0],target_graph_ids.size(1) - 3,target_graph_ids.size(1) - 3).to(torch.int)

        input_ids = torch.cat((text_ids,target_graph_ids),dim = -1)
        attention_mask_graph = torch.ones_like(target_graph_ids)
        attention_mask = torch.cat((attention_mask_text,attention_mask_graph),dim = -1)

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
                batch["input_ids"], special_tokens_mask=special_tokens_mask, types='node', target_start = text_ids.size(1) + 1,
            )
            num_node = batch["edge_input"].shape[1]
            batch["edge_input"], batch["edge_labels"] = self.torch_mask_tokens(
                batch["edge_input"].reshape(batch["edge_input"].shape[0],-1), special_tokens_mask=special_tokens_mask, types='edge'
            )
            batch["edge_input"] = batch["edge_input"].reshape(batch["edge_input"].shape[0], num_node, num_node)
            batch["edge_labels"] = batch["edge_labels"].reshape(batch["edge_labels"].shape[0], num_node, num_node)

        return batch

class CollatorForMixedLanguageModeling(DataCollatorForLanguageModeling):

    def __init__(self,**kwargs):
        self.num_edges_type = kwargs.pop('num_edges_type')
        super().__init__(**kwargs)

    def __post_init__(self):

        if self.tf_experimental_compile:
            import tensorflow as tf
            self.tf_mask_tokens = tf.function(self.tf_mask_tokens, jit_compile=True)

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None, types='node', target_start = None) -> Tuple[Any, Any]:
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
                special_tokens_mask = [i[:target_start] + [0]*(len(i)-target_start) for i in special_tokens_mask]
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
            inputs[indices_replaced] = self.num_edges_type

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
        text_ids = pad_sequence([torch.tensor(d['input_ids']) if d['type'] == 'text' else torch.tensor([0]) for d in examples], batch_first=True)
        graph_ids = pad_sequence([torch.tensor(d['input_ids']) if d['type'] == 'graph' else torch.tensor([0]) for d in examples], batch_first=True)

        input_ids = torch.cat((text_ids,graph_ids),dim=-1)
        attention_mask_text = torch.ones_like(text_ids)
        attention_mask_text[text_ids == 0] = 0
        attention_mask_graph = torch.ones_like(graph_ids)
        attention_mask = torch.cat((attention_mask_text,attention_mask_graph),dim = -1)

        edge_input = torch.zeros(graph_ids.shape[0],graph_ids.size(1) - 2,graph_ids.size(1) - 2).to(torch.int)
        for i,d in enumerate(examples):
            if d['edge'] is not None:
                for (start,end,bond) in d['edge']:
                    edge_input[i,start,end] = bond
                    edge_input[i,end,start] = bond

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'edge_input' : edge_input,
            'target_start': text_ids.size(1) + 1
        }
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask, types='node', target_start = batch["target_start"],
            )
            num_node = batch["edge_input"].shape[1]
            batch["edge_input"], batch["edge_labels"] = self.torch_mask_tokens(
                batch["edge_input"].reshape(batch["edge_input"].shape[0],-1), special_tokens_mask=special_tokens_mask, types='edge'
            )
            batch["edge_input"] = batch["edge_input"].reshape(batch["edge_input"].shape[0], num_node, num_node)
            batch["edge_labels"] = batch["edge_labels"].reshape(batch["edge_labels"].shape[0], num_node, num_node)

        return batch
        

@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int

    def pad_examples(self, examples, padding_value=0):
        inputs = {k: [example[k] for example in examples] for k in examples[0].keys()}
        padded_inputs = padding(inputs, padding_value=padding_value)
        return padded_inputs

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input

        input_ids = pad_sequence([torch.tensor(d['group_input_ids']) for d in examples], batch_first=True)

        batch = BatchEncoding({
            'input_ids': input_ids,
        })

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
        
        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        ''''''
        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )
        
        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
