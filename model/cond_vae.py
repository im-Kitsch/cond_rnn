import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.traj_dataset_general import wrap_data
from model.base_vae import BaseVAE
from util.nlp_util import trans_one_hot, multinomial_choose


class CondVAEModel(torch.nn.Module):
    def __init__(self, num_class, embedding_size, embedding_drop_out,
                 hid_size, hid_layers, latent_z_size, cond_size, word_dropout_rate,
                 gru_drop_out, sos_idx, unk_idx):
        super(CondVAEModel, self).__init__()

        self.word_dropout_rate = word_dropout_rate
        self.latent_size = latent_z_size
        self.sos_idx = sos_idx
        self.cond_size = cond_size
        self.gru_layers = hid_layers
        self.gru_hid_size = hid_size

        self.emb = nn.Sequential(nn.Embedding(num_class, embedding_size),
                                 nn.Dropout(p=embedding_drop_out))
        self.enc_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)

        self.mean_z = torch.nn.Linear(hid_size * hid_layers, latent_z_size)
        self.logvar_z = torch.nn.Linear(hid_size * hid_layers, latent_z_size)
        self.latent2hidden = nn.Linear(latent_z_size+cond_size, hid_layers * hid_size)

        self.dec_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)
        self.output = nn.Linear(hid_size, num_class)
        return

    def forward(self, input_data, cond, z=None):
        seq_len, batch_size = input_data.shape
        if z is None:
            embeded = self.emb(input_data)
            _, hidden = self.enc_rnn(embeded)
            hidden = hidden.permute(1, 0, 2)  # (batch, layer*direction, hidden_size)
            hidden = hidden.reshape(batch_size, -1)

            mean = self.mean_z(hidden)
            log_var = self.logvar_z(hidden)
            std = torch.exp(0.5 * log_var)
            noise = torch.randn_like(mean)

            z = mean + noise * std     # (batch_size, z_size)
        else: # seems could never be used since inference
            embeded = self.emb(input_data)

        latent = torch.cat([z, cond.T], dim=1)
        hid_dec = self.latent2hidden(latent)
        hid_dec = hid_dec.view(batch_size, self.gru_layers, self.gru_hid_size)
        hid_dec = hid_dec.permute(1, 0, 2).contiguous()

        if self.word_dropout_rate != 0.:
            raise Exception("Not implemented")
        else:
            logits, _ = self.dec_rnn(embeded, hid_dec)

        logits = self.output(logits)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        return logits, mean, log_var

    def batch_forward(self, batch_data):
        data, label, cond = batch_data['data'], batch_data['label'], batch_data['cond']
        logits, mean, logvar = self.forward(data, cond)
        return logits, mean, logvar

    def sample_z(self, num, device=torch.device("cpu")):
        z = torch.randn(num, self.latent_size, device=device)
        return z

    def sample_cond(self, num, device=torch.device("cpu")):
        cond = np.random.randint(0, self.cond_size, num)
        cond = trans_one_hot(cond, self.cond_size)
        return torch.tensor(cond, dtype=torch.float32, device=device)

    # attention, z would be better for batch first
    def inference(self, z, cond, pred_length, input_data=None,
                  batch_first_z_cond=True, device=torch.device("cpu")):

        batch_size = z.shape[0] if batch_first_z_cond else z.shape[1]
        z = z if batch_first_z_cond else z.T
        cond = cond if batch_first_z_cond else cond.T
        if input_data is None:
            data = torch.zeros(1, batch_size, dtype=int, device=device) + self.sos_idx
        else:
            raise Exception("not implemented yet")

        pred_traj = torch.zeros(pred_length, batch_size, dtype=int, device=device)

        latent = torch.cat([z, cond], dim=1)
        hid_dec = self.latent2hidden(latent)
        hid_dec = hid_dec.view(batch_size, self.gru_layers, self.gru_hid_size)
        hid_dec = hid_dec.permute(1, 0, 2).contiguous()

        data_t = data
        for i in range(pred_length):
            embeded = self.emb(data_t)
            logits, hid_dec = self.dec_rnn(embeded, hid_dec)
            logits = self.output(logits)
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            logits = logits[[-1]] # keep dim and use only last one

            _, data_t = multinomial_choose(logits, topk=3)
            # data_t.to(device)
            pred_traj[i] = data_t.flatten()

        pred_traj = torch.cat([data, pred_traj], dim=0)

        return pred_traj


class CondVAE(BaseVAE):
    def __init__(self, num_class, embedding_size, embedding_drop_out, hid_size, hid_layers,
                 latent_z_size, cond_size, word_dropout_rate, gru_drop_out, sos_idx, unk_idx,
                 anneal_k, anneal_x0, anneal_function, data_keys, learning_rate):

        assert data_keys == ["data", "label", "cond"]

        model = CondVAEModel(num_class, embedding_size, embedding_drop_out,
                             hid_size, hid_layers, latent_z_size, cond_size,
                             word_dropout_rate, gru_drop_out, sos_idx, unk_idx)
        super(CondVAE, self).__init__(model=model, anneal_k=anneal_k, anneal_x0=anneal_x0,
                                      anneal_function=anneal_function, learning_rate=learning_rate,
                                      data_keys=data_keys, writer_comment="CondVAE")
        return
