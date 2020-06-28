import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.traj_dataset_general import wrap_data
from model.base_vae import BaseVAE
from util.nlp_util import trans_one_hot, multinomial_choose


def expand_input(input_data, cond_size):
    assert input_data.ndim == 2
    device = input_data.device
    seq_len, batch_size = input_data.shape
    input_data = input_data.reshape(seq_len, batch_size, 1)
    input_data = input_data.repeat(1, 1, cond_size)  # seq_len, batch_size, cond_size

    cond = trans_one_hot(np.arange(cond_size), cond_size)
    cond = torch.tensor(cond, dtype=torch.float32, device=device)
    cond = cond.reshape(1, cond_size, cond_size)
    cond = cond.repeat(batch_size, 1, 1)  # batch_size, cond_size, cond_size

    return input_data.reshape(seq_len, -1), cond.reshape(-1, cond_size)


class DisentangledVAEModel(torch.nn.Module):
    def __init__(self, num_class, embedding_size, embedding_drop_out,
                 hid_size, hid_layers, latent_z_size, cond_size, word_dropout_rate,
                 gru_drop_out, sos_idx, unk_idx):
        super(DisentangledVAEModel, self).__init__()

        self.word_dropout_rate = word_dropout_rate
        self.latent_z_size = latent_z_size
        self.num_class = num_class
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

        self.classify_rnn = torch.nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                                         num_layers=hid_layers, dropout=gru_drop_out,
                                         bias=True, bidirectional=True)
        self.classify_nn = nn.Linear(hid_size*2, cond_size)
        return

    def forward(self, input_data, cond=None, z=None, batch_first_cond=True):
        seq_len, batch_size = input_data.shape
        device = input_data.device
        if cond is None:
            input_data, cond = expand_input(input_data, self.cond_size)
            is_labeled = False
        else:
            cond = cond if batch_first_cond else cond.T
            is_labeled = True

        if z is None:  # TODO think the logic, order agiain
            mean, logvar, embeded = self._enc_forward(input_data)
        else:
            raise Exception("Not implemented")

        z = self.sample_z(mean=mean, logvar=logvar, device=device)
        hid_dec = self._latent2hid_dec(z=z, cond=cond, batch_first_z=True, batch_first_cond=True)
        logits, _ = self._dec_forward(input_data=None, input_embeded=embeded, hid_dec=hid_dec)

        classify_logits = self._classify_forward(input_data=None, input_embeded=embeded)
        if not is_labeled:
            logits = logits.reshape(seq_len, batch_size, self.cond_size, self.num_class)
            logits = logits.permute(2, 0, 1, 3)
            classify_logits = classify_logits.reshape(batch_size, self.cond_size, self.cond_size)
            classify_logits = classify_logits.permute(1, 0, 2)
            mean = mean.reshape(batch_size, self.cond_size, self.latent_z_size)
            mean = mean.permute(1, 0, 2)
            logvar = logvar.reshape(batch_size, self.cond_size, self.latent_z_size)
            logvar = logvar.permute(1, 0, 2)

        return logits, mean, logvar, classify_logits

    def _enc_forward(self, input_data):
        batch_size = input_data.shape[1]
        embeded = self.emb(input_data)
        _, hidden = self.enc_rnn(embeded)
        hidden = hidden.permute(1, 0, 2)  # (batch, layer*direction, hidden_size)
        hidden = hidden.reshape(batch_size, -1)

        mean = self.mean_z(hidden)
        log_var = self.logvar_z(hidden)
        return mean, log_var, embeded

    def sample_z(self, size=None, mean=None, logvar=None, device=torch.device("cpu")):
        if (mean is None) and (logvar is None):
            assert size is None
            z = torch.randn(size, device=device)
        elif size is None:
            assert not(mean is None)
            assert not(logvar is None)
            noise = torch.randn_like(mean, device=device)
            std = torch.exp(0.5 * logvar)
            z = mean + noise * std  # (batch_size, z_size)
        else:
            raise Exception("False")
        return z

    def sample_cond(self, size=None, y_dist=None, device=torch.device("cpu")):
        if size is None:
            raise Exception("Not implemented")
        elif y_dist is None:
            cond = np.random.randint(0, self.cond_size, size)
            cond = trans_one_hot(cond, self.cond_size)
            cond = torch.tensor(cond, dtype=torch.float32, device=device)
        return cond

    def _latent2hid_dec(self, z, cond, batch_first_z, batch_first_cond):
        z = z if batch_first_z else z.T
        cond = cond if batch_first_cond else cond.T
        batch_size = z.shape[0]

        latent = torch.cat([z, cond], dim=1)
        hid_dec = self.latent2hidden(latent)
        hid_dec = hid_dec.view(batch_size, self.gru_layers, self.gru_hid_size)
        hid_dec = hid_dec.permute(1, 0, 2).contiguous()
        return hid_dec

    def _dec_forward(self, input_data, input_embeded, hid_dec):
        assert input_data is None or input_embeded is None

        if self.word_dropout_rate != 0.:
            raise Exception("Not implemented")
        else:
            assert input_data is None
            assert not(input_embeded is None)
            logits, hid_dec = self.dec_rnn(input_embeded, hid_dec)

        logits = self.output(logits)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        return logits, hid_dec

    def _classify_forward(self, input_data, input_embeded):
        assert input_data is None or input_embeded is None
        if self.word_dropout_rate != 0.:
            raise Exception("Not implemented")
        else:
            assert input_data is None
            assert not(input_embeded is None)
            logits, _ = self.classify_rnn(input_embeded)

        logits = self.classify_nn(logits)
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        return logits[-1]


class DisentangledVAE(BaseVAE):
    def __init__(self, num_class, embedding_size, embedding_drop_out, hid_size, hid_layers,
                 latent_z_size, cond_size, word_dropout_rate, gru_drop_out, sos_idx, unk_idx,
                 anneal_k, anneal_x0, anneal_function, data_keys, learning_rate):

        assert data_keys == ["data", "label", "cond"]
        model = DisentangledVAEModel(num_class, embedding_size, embedding_drop_out,
                                hid_size, hid_layers, latent_z_size, cond_size, word_dropout_rate,
                                gru_drop_out, sos_idx, unk_idx)
        super(DisentangledVAE, self).__init__(model=model, anneal_k=anneal_k, anneal_x0=anneal_x0,
                                              anneal_function=anneal_function, learning_rate=learning_rate,
                                              data_keys=data_keys, writer_comment="DisentangledVAE")
        return

    def train_step(self, batch_data):
        data, cond, label =batch_data["data"], batch_data["cond"], batch_data["label"]
        batch_size_labeled = int(data.shape[1] * 0.5)

        sub_data, sub_cond, sub_label = data[:, :batch_size_labeled], cond[:, :batch_size_labeled], label[:, :batch_size_labeled]
        logits, mean, logvar, classify_logits = self.model.forward(input_data=sub_data, cond=sub_cond.T,
                                                                   batch_first_cond=True)
        loss1 = self.disentangled_loss(logits=logits, label=sub_label, mean=mean, logvar=logvar,
                                       classify_logits=classify_logits, cond=sub_cond.T, batchfirst_cond=True)

        sub_data, sub_label = data[:, batch_size_labeled:], label[:, batch_size_labeled:]
        logits, mean, logvar, classify_logits = self.model.forward(input_data=sub_data, cond=None)
        loss2 = self.disentangled_loss(logits=logits, label=sub_label, mean=mean, logvar=logvar,
                                       classify_logits=classify_logits, cond=None, batchfirst_cond=None)

        loss = loss1 + loss2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, loss1, loss2, 1

    @staticmethod
    def  disentangled_loss(logits, label, mean, logvar, classify_logits, cond, batchfirst_cond):

        # logits cond_size, seq_len, batch_size, num_class
        if logits.ndim == 4:
            cond_size, seq_len, batch_size, num_class = logits.shape
            assert label.shape == torch.Size([seq_len, batch_size])
            # assert cond.shape == torch.Size([batch_size, cond_size])
            assert classify_logits.shape == torch.Size([cond_size, batch_size, cond_size])
            assert cond is None

            logits = logits.permute(0, 3, 1, 2) # cond_size, num_class, seq_len, batch_size
            label = label.unsqueeze(0).repeat(cond_size, 1, 1)
            nll_loss = F.nll_loss(logits, label, reduction="none")
            nll_loss = nll_loss.sum(dim=1)

            cond_labeled = torch.arange(cond_size, device=logits.device).reshape(-1, 1).repeat(1, batch_size)
            # cond_size, batch_size
            # _, cond_labeled = torch.max(cond, dim=1)
            # cond_labeled = cond_labeled.reshape(1, batch_size).repeat(cond_size, 1)
            classify_logits = classify_logits.permute(0, 2, 1)
            classify_loss = F.nll_loss(classify_logits, cond_labeled, reduction="none")
            p_classify = torch.exp(-classify_loss)

            loss_kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
            loss_kl = loss_kl.sum(dim=2)

            loss = p_classify * (nll_loss + loss_kl + classify_loss)
            loss = loss.sum(dim=0)
            loss = loss.mean()

        elif logits.ndim == 3:
            cond = cond if batchfirst_cond else cond.T
            seq_len, batch_size, num_class = logits.shape
            _, cond_size = cond.shape
            assert label.shape == torch.Size([seq_len, batch_size])
            assert cond.shape == torch.Size([batch_size, cond_size])
            assert classify_logits.shape == torch.Size([batch_size, cond_size])

            loss_kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
            loss_kl = loss_kl.sum(dim=1)

            logits = logits.permute(0, 2, 1)
            loss_nll = F.nll_loss(logits, label, reduction="none")
            loss_nll = loss_nll.sum(dim=0)

            _, cond_labeled = torch.max(cond, dim=1)
            cond_labeled = cond_labeled.flatten()
            classify_loss = F.nll_loss(classify_logits, cond_labeled, reduction="none")
            loss = loss_nll + loss_kl + classify_loss
            loss = loss.mean()
        else:
            raise Exception("False para")
        return loss

    @torch.no_grad()
    def eval_metric(self, data_loader, device):
        self.model.eval()

        total_ppl = 0
        total_elbo = 0
        total_samples = 0
        weight = 1.
        for i_bt, batch_data in enumerate(data_loader):
            batch_data = wrap_data(batch_data, self.data_keys, device=device)
            data, cond, label = batch_data["data"], batch_data["cond"], batch_data["label"]
            batch_size = label.shape[1]

            logits, mean, logvar, classify_logits = self.model.forward(input_data=data,
                                                                       cond=cond.T, z=None, batch_first_cond=True)
            loss = self.disentangled_loss(logits, label, mean, logvar, classify_logits,
                                                       cond.T, batchfirst_cond=True)

            # loss = loss_nll + loss_kl * weight

            ppl = torch.nn.functional.nll_loss(logits.permute(0, 2, 1), label, reduction="mean")
            # NLL_loss is setting as reduction = "none"

            total_elbo += loss.item() * batch_size
            total_ppl += ppl.item() * batch_size
            total_samples += batch_size

        return total_elbo / total_samples, np.exp(total_ppl / total_samples)

