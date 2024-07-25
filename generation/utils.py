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


def tokenize_function_gimlet(examples, tokenizer, text_column_name, padding, max_seq_length):

    text = tokenizer(
        #examples[text_column_name] + str(examples['label']) if isinstance(examples[text_column_name],str) else examples[text_column_name][0] + str(examples['label']),
        str(examples['label']),
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False
    )

    graph_data = examples['graph']

    return {'graph': graph_data,
            'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask']}

def tokenize_function_mixed(examples, tokenizer, padding, max_seq_length):
    if examples['type'] == 'text':
        text = tokenizer(
            examples['text'],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False
        )
        return {'input_ids': text.data['input_ids'],
                'attention_mask': text.data['attention_mask'],
                'edge': None
            }
    if examples['type'] == 'graph':
        graph_ids, graph_edge = graph_collect(examples['text'],tokenizer)
        #print(graph_ids, graph_edge)
        return {'input_ids': graph_ids,
                'attention_mask': None,
                'edge': graph_edge
            }
    
def tokenize_function_t5(examples, tokenizer, padding, max_seq_length):
    text = tokenizer(
        examples['text'],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False
    )
    return {'input_ids': text.data['input_ids'],
            'attention_mask': text.data['attention_mask'],
            'edge': None
        }

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

