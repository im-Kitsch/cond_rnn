#
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

SEED = 512

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


from util.data_prepare import traj_prepare, cond_seq_prepare
from util.traj_dataset_general import TrajDatasetGeneral
from util.nlp_util import seq_length_from_padded_seq
import numpy as np

BATCH_SIZE = 128

num_samples = 2e4
source_folder = "/home/ubuntu/pflow_data/filtered"
traj_path = f"{source_folder}/traj.csv"
code_path = f"{source_folder}/code.csv"
cond_seq_path = f"{source_folder}/cond_seq_pad.csv"

traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                        num_samples, use_cols=360+np.arange(18*3)*20, add_idx=["0/SOS"])
cond_seq, n_cond_seq, cond_pad_idx = cond_seq_prepare(cond_seq_path, num_samples)

cond_seq_len = seq_length_from_padded_seq(cond_seq, batch_first=True, padded_idx=cond_pad_idx)
mask = cond_seq_len > 4
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
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src = [src len, batch size]
        # src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        max_src_len = src_len.max()
        src = src[:max_src_len]

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        attention_weights = []
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _attn = self.decoder(input, hidden, encoder_outputs, mask)
            attention_weights.append(_attn.unsqueeze(1))

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        attention_weights = torch.cat(attention_weights, dim=1)
        return outputs, attention_weights


INPUT_DIM = n_cond_seq
OUTPUT_DIM = n_traj
ENC_EMB_DIM = 16
DEC_EMB_DIM = 64
ENC_HID_DIM = 16
DEC_HID_DIM = 64
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
SRC_PAD_IDX = cond_pad_idx

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
device = torch.device("cuda:0")
model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=5e-4)



criterion = nn.CrossEntropyLoss()


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg, src_len = batch
        src = src.T.to(model.device)
        trg = trg.T.to(model.device)
        src_len = src_len.to(model.device).flatten()

        optimizer.zero_grad()

        output, _ = model(src, src_len, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg, src_len = batch
            src = src.T.to(model.device)
            trg = trg.T.to(model.device)
            src_len = src_len.to(model.device).flatten()

            output, _ = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
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

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'attention-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


model.load_state_dict(torch.load('attention-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


# def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
#     model.eval()
#
#     if isinstance(sentence, str):
#         nlp = spacy.load('de')
#         tokens = [token.text.lower() for token in nlp(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]
#
#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#
#     src_indexes = [src_field.vocab.stoi[token] for token in tokens]
#
#     src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
#
#     src_len = torch.LongTensor([len(src_indexes)]).to(device)
#
#     with torch.no_grad():
#         encoder_outputs, hidden = model.encoder(src_tensor, src_len)
#
#     mask = model.create_mask(src_tensor)
#
#     trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
#
#     attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
#
#     for i in range(max_len):
#
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
#
#         with torch.no_grad():
#             output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)
#
#         attentions[i] = attention
#
#         pred_token = output.argmax(1).item()
#
#         trg_indexes.append(pred_token)
#
#         if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
#             break
#
#     trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
#
#     return trg_tokens[1:], attentions[:len(trg_tokens) - 1]
#
#
# def display_attention(sentence, translation, attention):
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111)
#
#     attention = attention.squeeze(1).cpu().detach().numpy()
#
#     cax = ax.matshow(attention, cmap='bone')
#
#     ax.tick_params(labelsize=15)
#     ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
#                        rotation=45)
#     ax.set_yticklabels([''] + translation)
#
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#     plt.close()

# example_idx = 12
#
# src = vars(train_data.examples[example_idx])['src']
# trg = vars(train_data.examples[example_idx])['trg']
#
# print(f'src = {src}')
# print(f'trg = {trg}')
#
# translation, attention = translate_sentence(src, SRC, TRG, model, device)
#
# print(f'predicted trg = {translation}')
#
# display_attention(src, translation, attention)
