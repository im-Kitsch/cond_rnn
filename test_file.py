
#
# ######################################################################
# # Now that we've defined ``train_data``, we can see an extremely useful
# # feature of ``torchtext``'s ``Field``: the ``build_vocab`` method
# # now allows us to create the vocabulary associated with each language
#
# SRC.build_vocab(train_data, min_freq = 2)
# TRG.build_vocab(train_data, min_freq = 2)

######################################################################
# Once these lines of code have been run, ``SRC.vocab.stoi`` will  be a
# dictionary with the tokens in the vocabulary as keys and their
# corresponding indices as values; ``SRC.vocab.itos`` will be the same
# dictionary with the keys and values swapped. We won't make extensive
# use of this fact in this tutorial, but this will likely be useful in
# other NLP tasks you'll encounter.

######################################################################
# ``BucketIterator``
# ----------------
# The last ``torchtext`` specific feature we'll use is the ``BucketIterator``,
# which is easy to use since it takes a ``TranslationDataset`` as its
# first argument. Specifically, as the docs say:
# Defines an iterator that batches examples of similar lengths together.
# Minimizes amount of padding needed while producing freshly shuffled
# batches for each new epoch. See pool for the bucketing procedure used.

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1024

# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data, valid_data, test_data),
#     batch_size = BATCH_SIZE,
#     device = device)

import torch
from util.data_prepare import traj_prepare
from util.traj_dataset_general import TrajDatasetGeneral
import numpy as np

num_samples = 20000
source_folder = "/home/ubuntu/pflow_data/filtered"
traj_path = f"{source_folder}/traj.csv"
code_path = f"{source_folder}/code.csv"
traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                        num_samples, use_cols=360+np.arange(18*6)*10, add_idx=["0/SOS"])
ix_SOS = translator.code2ix.loc["0/SOS"].item()
src, trg = traj[:, :50], traj[:, 50:]
trg = np.concatenate([np.zeros((trg.shape[0], 1), dtype=int)+ix_SOS, trg], axis=1)

training_dataset = TrajDatasetGeneral([src[:15000], trg[:15000]])
validation_dataset = TrajDatasetGeneral([src[15000:18000], trg[15000:18000]])
test_dataset = TrajDatasetGeneral([src[18000:], trg[18000:]])

train_iterator = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
#%%



######################################################################
# These iterators can be called just like ``DataLoader``s; below, in
# the ``train`` and ``evaluate`` functions, they are called simply with:
#
# ::
#
#    for i, batch in enumerate(iterator):
#
# Each ``batch`` then has ``src`` and ``trg`` attributes:
#
# ::
#
#    src = batch.src
#    trg = batch.trg

######################################################################

# Note: this model is just an example model that can be used for language
# translation; we choose it because it is a standard model for the task,
# not because it is the recommended model to use for translation. As you're
# likely aware, state-of-the-art models are currently based on Transformers;
# you can see PyTorch's capabilities for implementing Transformer layers
# `here <https://pytorch.org/docs/stable/nn.html#transformer-layers>`__; and
# in particular, the "attention" used in the model below is different from
# the multi-headed self-attention present in a transformer model.


import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src: Tensor) -> Tuple[Tensor]:

        embedded = self.dropout(self.embedding(src))

        outputs, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 attn_dim: int):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim

        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tensor:

        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim = 2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)


    def _weighted_encoder_rep(self,
                              decoder_hidden: Tensor,
                              encoder_outputs: Tensor) -> Tensor:

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep, a.detach()


    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        weighted_encoder_rep, att_weights = self._weighted_encoder_rep(decoder_hidden,
                                                          encoder_outputs)

        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim = 2)

        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)

        output = self.out(torch.cat((output,
                                     weighted_encoder_rep,
                                     embedded), dim = 1))

        return output, decoder_hidden.squeeze(0), att_weights


class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                src: Tensor,
                trg: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> token
        output = trg[0,:]

        att_weights = []
        for t in range(1, max_len):
            output, hidden, att_wei = self.decoder(output, hidden, encoder_outputs)
            att_weights.append(att_wei)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        att_weights = torch.cat(att_weights, dim=1)
        return outputs, att_weights


INPUT_DIM = n_traj
OUTPUT_DIM = n_traj
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# ENC_HID_DIM = 512
# DEC_HID_DIM = 512
# ATTN_DIM = 64
# ENC_DROPOUT = 0.5
# DEC_DROPOUT = 0.5

ENC_EMB_DIM = 32
DEC_EMB_DIM = 32
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)

attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

######################################################################
# Note: when scoring the performance of a language translation model in
# particular, we have to tell the ``nn.CrossEntropyLoss`` function to
# ignore the indices where the target is simply padding.

# PAD_IDX = TRG.vocab.stoi['<pad>']

# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion = nn.CrossEntropyLoss()
######################################################################
# Finally, we can train and evaluate this model:

import math
import time


def train(model: nn.Module,
          iterator: torch.utils.data.dataloader.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, batch in enumerate(iterator):

        src, trg = batch
        src = src.T.to(model.device)
        trg = trg.T.to(model.device)

        optimizer.zero_grad()

        output, att_weights = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.dataloader.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):
            src, trg = batch
            src = src.T.to(model.device)
            trg = trg.T.to(model.device)

            output, att_weights = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 20
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


