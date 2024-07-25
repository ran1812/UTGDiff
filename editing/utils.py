import torch
import numpy as np


def fori_loop(lower, upper, body_fun, init_val):
    val = init_val

    for i in range(lower, upper):
        val = body_fun(i, val)

    return val


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, torch.stack(ys) if ys[0] is not None else None


def word_frequency(path, basic_freq=.5):
    freq = torch.load(path)
    return basic_freq + (1 - basic_freq) * freq / freq.mean()

def min_max_norm(t, dim):
    return ((t - t.min(dim=dim, keepdims=True).values) / (t.max(dim=dim, keepdims=True).values - t.min(dim=dim, keepdims=True).values)) * 2 - 1

def is_symmetric(matrix,dim1=0,dim2=1):
    return torch.allclose(matrix, matrix.transpose(dim1,dim2))

def trans_into_sym_from_triu(edge_matrix):
    if edge_matrix.dim() == 3:
        edge_matrix = (edge_matrix.triu(1) + edge_matrix.transpose(1,2).tril(0))
    if edge_matrix.dim() == 4:
        edge_matrix = (edge_matrix.permute(0,3,1,2).triu(1) + edge_matrix.permute(0,3,1,2).transpose(2,3).tril(0)).permute(0,2,3,1)
    return edge_matrix

def return_nonzero_index(matrix):
    nonzero_indices = matrix.nonzero()
    if matrix.dim() == 2:
        nonzero_values = matrix[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    if matrix.dim() == 3:
        nonzero_values = matrix[nonzero_indices[:, 0], nonzero_indices[:, 1], nonzero_indices[:, 2]]
    result = torch.cat((nonzero_indices, nonzero_values.unsqueeze(1)), dim=1)
    return result


def tokenize_function_gimlet(examples, tokenizer, text_column_name, padding, max_seq_length, rich_features, transform_in_collator):

    text = tokenizer(
        examples[text_column_name] + str(examples['label']) if isinstance(examples[text_column_name],str) else examples[text_column_name][0] + str(examples['label']),
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
        # receives the `special_tokens_mask`.
        add_special_tokens=False
    )

    graph_data = examples['graph']

    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask']}
