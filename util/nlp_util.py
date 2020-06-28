import torch
import numpy as np


def seq_length_from_padded_seq(src, batch_first, padded_idx):
    assert src.ndim == 2
    seq_dim = 1 if batch_first else 0
    seq_max_len = src.shape[seq_dim]
    pad_len = (src == padded_idx).sum(seq_dim)

    return seq_max_len - pad_len


def multinomial_choose(logits, topk, temperature=1.):
    assert logits.ndim == 3

    topk_val, topk_ix = torch.topk(logits, topk, dim=2)

    n_r, n_c = logits.shape[0], logits.shape[1]

    topk_val /= temperature
    topk_val = torch.exp(topk_val)
    multinomial_ix = torch.multinomial(topk_val.reshape(n_r*n_c, -1), 1)
    multinomial_ix = multinomial_ix.reshape(n_r, n_c, 1)

    choice_val = torch.gather(topk_val, 2, multinomial_ix)
    choice_ix = torch.gather(topk_ix, 2, multinomial_ix)
    choice_val, choice_ix = choice_val.squeeze(2), choice_ix.squeeze(2)

    return choice_val, choice_ix


def trans_one_hot(data, n_categ):
    if type(data) is list:
        one_hot_code = np.eye(n_categ)[data]
    elif type(data) is np.ndarray:

        data = np.expand_dims(data, axis=-1)
        orig_shape = list(data.shape)
        orig_shape[-1] = n_categ

        one_hot_code = np.eye(n_categ)[data.flatten()]
        one_hot_code = one_hot_code.reshape(orig_shape)
    return one_hot_code


