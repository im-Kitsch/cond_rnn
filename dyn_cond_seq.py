
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 512

from util.data_prepare import traj_prepare, cond_seq_prepare
from util.traj_dataset_general import TrajDatasetGeneral
from util.nlp_util import seq_length_from_padded_seq
import numpy as np

num_samples = 50000
source_folder = "/home/ubuntu/pflow_data/filtered"
traj_path = f"{source_folder}/traj.csv"
code_path = f"{source_folder}/code.csv"
cond_seq_path = f"{source_folder}/cond_seq_pad.csv"

traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                        num_samples, use_cols=360+np.arange(18*2)*30, add_idx=["0/SOS"])
cond_seq, n_cond_seq, cond_pad_idx = cond_seq_prepare(cond_seq_path, num_samples)

cond_seq_len = seq_length_from_padded_seq(cond_seq, batch_first=True, padded_idx=cond_pad_idx)
mask = cond_seq_len > 1
traj = traj[mask]
cond_seq = cond_seq[mask]
cond_seq_len = cond_seq_len[mask].reshape(-1, 1)
num_samples = traj.shape[0]
num_train = int(num_samples*0.6)
num_validation = int(num_samples*0.2)

ix_SOS = translator.code2ix.loc["0/SOS"].item()
src, trg = cond_seq, traj
trg = np.concatenate([np.zeros((trg.shape[0], 1), dtype=int)+ix_SOS, trg], axis=1)

training_dataset = TrajDatasetGeneral([src[:num_train], trg[:num_train], cond_seq_len[:num_train]])
validation_dataset = TrajDatasetGeneral([src[num_train:num_train+num_validation],
                                         trg[num_train:num_train+num_validation],
                                         cond_seq_len[num_train:num_train+num_validation]])
test_dataset = TrajDatasetGeneral([src[num_train+num_validation:],
                                   trg[num_train+num_validation:],
                                   cond_seq_len[num_train+num_validation:]])

train_iterator = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
#%%


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
                 dropout: float,
                 pad_idx:int):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

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

    def single_forward(self,
                src: Tensor,
                trg: Tensor,
                src_len: Tensor,
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

            output, hidden, att_wei = self.decoder(output, hidden, encoder_outputs[:src_len[0]])
            att_weights.append(att_wei)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        att_weights = torch.cat(att_weights, dim=1)
        return outputs, att_weights.squeeze(0)

    def forward(self, src, trg, src_len, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        outputs, att_weights = [], []
        for i in range(batch_size):
            output, att_wei = self.single_forward(src[:, [i]],
                                                  trg[:, [i]],
                                                  src_len[[i]],
                                                  teacher_forcing_ratio)
            outputs.append(output)
            att_weights.append(att_wei)

        outputs = torch.cat(outputs, dim=1)
        return outputs.contiguous(), att_weights


INPUT_DIM = n_cond_seq
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
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, pad_idx=cond_pad_idx)

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

if not (model.encoder.embedding.padding_idx is None):
    nn.init.zeros_(model.encoder.embedding.weight[0])

optimizer = optim.Adam(model.parameters(), lr=5e-3)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')


# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion = nn.CrossEntropyLoss()
######################################################################
#%%
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

    for _i, batch in enumerate(iterator):

        src, trg, src_len = batch
        src = src.T.to(model.device)
        trg = trg.T.to(model.device)
        src_len = src_len.to(model.device)

        optimizer.zero_grad()

        output, att_weights = model(src, trg, src_len=src_len)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        print(_i, loss.item())

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: torch.utils.data.dataloader.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, batch in enumerate(iterator):
            src, trg, src_len = batch
            src = src.T.to(model.device)
            trg = trg.T.to(model.device)
            src_len = src_len.to(model.device)

            output, att_weights = model(src, trg, src_len=src_len, teacher_forcing_ratio=0.) #turn off teacher forcing

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


N_EPOCHS = 10
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


