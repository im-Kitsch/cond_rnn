import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.traj_dataset_general import wrap_data
from util.nlp_util import multinomial_choose
from model.base_vae import BaseVAE


class SeqVAEModel(torch.nn.Module):
    def __init__(self, num_class, embedding_size, embedding_drop_out,
                 hid_size, hid_layers, latent_z_size, word_dropout_rate,
                 gru_drop_out, sos_idx, unk_idx):
        super(SeqVAEModel, self).__init__()

        self.word_dropout_rate = word_dropout_rate
        self.latent_size = latent_z_size
        self.gru_layers = hid_layers
        self.gru_hid_size = hid_size
        self.sos_idx = sos_idx

        self.emb = nn.Sequential(nn.Embedding(num_class, embedding_size),
                                 nn.Dropout(p=embedding_drop_out))
        self.enc_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)

        self.mean_z = torch.nn.Linear(hid_size * hid_layers, latent_z_size)
        self.logvar_z = torch.nn.Linear(hid_size * hid_layers, latent_z_size)
        self.latent2hidden = nn.Linear(latent_z_size, hid_layers * hid_size)

        self.dec_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)
        self.output = nn.Linear(hid_size, num_class)
        return

    # z should be the same shape with batch size
    def forward(self, input_data, z=None):
        assert (z is None) or (dec_h is None)
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

            z = mean + noise * std
        else:
            assert z.shape[0] == batch_size
            embeded = self.emb(input_data)

            mean, log_var = None, None

        hid_dec = self.latent2hidden(z)
        hid_dec = hid_dec.view(batch_size, self.gru_layers, self.gru_hid_size)
        hid_dec = hid_dec.permute(1, 0, 2).contiguous()

        if self.word_dropout_rate != 0.:
            raise Exception("Not implemented")
        else:
            logits, _ = self.dec_rnn(embeded, hid_dec)

        logits = self.output(logits)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)

        return logits, mean, log_var

    # TODO not used here
    def batch_forward(self, batch_data, z=None):
        data, label = batch_data['data'], batch_data['label']
        logits, mean, logvar = self.forward(data)
        return logits, mean, logvar

    # no beam search
    @torch.no_grad()
    def inference(self, z, pred_length, batch_data=None, batch_first_z=True, device=torch.device("cpu")):
        batch_size = z.shape[0] if batch_first_z else z.shape[1]
        z = z if batch_first_z else z.T
        if batch_data is None:
            data = torch.zeros(1, batch_size, dtype=int, device=device) + self.sos_idx
        else:
            raise Exception("not implemented")

        pred_traj = torch.zeros(pred_length, batch_size, dtype=int, device=device)

        hid_dec = self.latent2hidden(z)
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

    def sample_z(self, num, device=torch.device("cpu")):
        z = torch.randn(num, self.latent_size, device=device)
        return z


class SeqVAE(BaseVAE):
    def __init__(self, num_class, embedding_size, embedding_drop_out, hid_size, hid_layers,
                 latent_z_size, word_dropout_rate, gru_drop_out, sos_idx, unk_idx,
                 anneal_k, anneal_x0, anneal_function, data_keys, learning_rate):

        assert data_keys == ["data", "label"]

        model = SeqVAEModel(num_class, embedding_size, embedding_drop_out,
                            hid_size, hid_layers, latent_z_size, word_dropout_rate,
                            gru_drop_out, sos_idx, unk_idx)
        super(SeqVAE, self).__init__(model=model, anneal_k=anneal_k, anneal_x0=anneal_x0,
                                     anneal_function=anneal_function, learning_rate=learning_rate,
                                     data_keys=data_keys, writer_comment="SeqVAE")
        return

