import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.traj_dataset_general import wrap_data


# non-bidirectional
class SeqVAEModel(torch.nn.Module):
    def __init__(self, num_class, embedding_size, embedding_drop_out,
                 hid_size, hid_layers, latent_z_size, word_dropout_rate,
                 gru_drop_out, sos_idx, unk_idx):
        super(SeqVAEModel, self).__init__()

        self.word_dropout_rate = word_dropout_rate
        self.latent_size = latent_z_size
        self.gru_layers = hid_layers
        self.gru_hid_size = hid_size

        self.emb = nn.Sequential(nn.Embedding(num_class, embedding_size),
                                 nn.Dropout(p=embedding_drop_out))
        self.enc_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)
        
        self.mean_z = torch.nn.Linear(hid_size*hid_layers, latent_z_size)
        self.logvar_z = torch.nn.Linear(hid_size*hid_layers, latent_z_size)
        self.latent2hidden = nn.Linear(latent_z_size, hid_layers*hid_size)

        self.dec_rnn = nn.GRU(input_size=embedding_size, hidden_size=hid_size,
                              num_layers=hid_layers, dropout=gru_drop_out,
                              bias=True, bidirectional=False)
        self.output = nn.Linear(hid_size, num_class)
        return

    def forward(self, input_data):
        seq_len, batch_size = input_data.shape
        embeded = self.emb(input_data)
        _, hidden = self.enc_rnn(embeded)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(batch_size, -1)

        mean = self.mean_z(hidden)
        log_var = self.logvar_z(hidden)
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(mean)

        z = mean + noise * std
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


class Annealer:
    def __init__(self, k, x0, anneal_function):
        self.k = k
        self.x0 = x0
        self._step_count = 0.
        self.anneal_function = anneal_function
        return

    def step(self, count=True):
        if self.anneal_function == 'logistic':
            weight = float(1/(1+np.exp(-self.k*(self._step_count - self.x0))))
        elif self.anneal_function == 'linear':
            weight = min(1, self._step_count/self.x0)
        else:
            raise Exception("Not implemented")
        if count is True:
            self._step_count += 1
        return weight

    @property
    def weight(self):
        return self.step(count=False)

    @property
    def step_count(self):
        return self._step_count


class SeqVAE:
    def __init__(self, num_class, embedding_size, embedding_drop_out,
                 hid_size, hid_layers, latent_z_size, word_dropout_rate, gru_drop_out,
                 anneal_k, anneal_x0, anneal_function, data_keys,
                 learning_rate, writer_comment,
                 sos_idx, unk_idx):
        assert data_keys == ["data", "label"]

        self.data_keys = data_keys

        self.model = SeqVAEModel(num_class, embedding_size, embedding_drop_out,
                                 hid_size, hid_layers, latent_z_size, word_dropout_rate,
                                 gru_drop_out, sos_idx, unk_idx)
        self.nll_loss = nn.NLLLoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(comment=writer_comment)
        self.annealer = Annealer(k=anneal_k, x0=anneal_x0, anneal_function=anneal_function)
        return

    def loss(self, logits, label, mean, log_var):
        batch_size = label.shape[1]
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        logits = logits.permute(0, 2, 1)

        loss_nll = self.nll_loss(logits, label)
        loss_nll = loss_nll.sum(dim=0)

        loss_kl = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp())
        loss_kl = loss_kl.sum(dim=0)
        return loss_nll.mean(), loss_kl.mean()

    def fit(self, training_loader, validation_loader, epochs, device):
        total_step = 0
        for _epoc in range(epochs):

            with tqdm(total=len(training_loader), desc=f"{_epoc:2.0f}", ncols=100) as pbar:
                for i_bt, batch_data in enumerate(training_loader):
                    batch_data = wrap_data(batch_data, self.data_keys, device=device)

                    loss, loss_nll, loss_kl, weight = self.train_step(batch_data)

                    pbar.set_postfix({"loss": f"{loss:.3f}", "nll": f"{loss_nll:.2f}",
                                      "kl": f"{loss_kl:.2f}", "weight": f"{weight:.3f}"})
                    pbar.update()

                    total_step += 1
                    self.writer.add_scalar("training_loss", loss, total_step)
                    self.writer.add_scalar("training_loss_nll", loss_nll, total_step)
                    self.writer.add_scalar("training_loss_kl", loss_kl, total_step)
                    self.writer.add_scalar("annealing_weight", weight, total_step)

                pbar.close()
                training_elbo, training_ppl = self.eval_metric(training_loader, device)
                validation_elbo, validation_ppl = self.eval_metric(validation_loader, device)

                print(f"evaluation: ppl: {training_ppl:.1f}, val_ppl: {validation_ppl:.1f}, elbo: {training_elbo:.1f} val elbo: {validation_elbo:.1f}")
                # pbar.set_postfix({"ppl": f"{training_ppl:.1f}", "val_ppl": f"{validation_ppl:.1f}",
                #                   "elbo": f"{training_elbo:.1f}", "val_elbo": f"{validation_elbo:.1f}"})
                self.writer.add_scalar("ppl/train", training_ppl, _epoc)
                self.writer.add_scalar("ppl/val", validation_ppl, _epoc)
                self.writer.add_scalar("elbo/train", training_elbo, _epoc)
                self.writer.add_scalar("elbo/val", validation_elbo, _epoc)
        return

    def train_step(self, batch_data):
        data, label = batch_data["data"], batch_data["label"]
        logits, mean, logvar = self.model(data)
        loss_nll, loss_kl = self.loss(logits, label, mean, logvar)
        weight = self.annealer.step()
        loss = loss_nll + loss_kl * weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_nll.item(), loss_kl.item(), weight

    @torch.no_grad()
    def eval_metric(self, data_loader, device):
        total_ppl = 0
        total_elbo = 0
        total_samples = 0
        weight = self.annealer.weight
        for i_bt, batch_data in enumerate(data_loader):
            batch_data = wrap_data(batch_data, self.data_keys, device=device)
            data, label = batch_data["data"], batch_data["label"]
            batch_size = data.shape[1]

            logits, mean, logvar = self.model(data)
            loss_nll, loss_kl = self.loss(logits, label, mean, logvar)

            loss = loss_nll + loss_kl * weight

            ppl = torch.nn.functional.nll_loss(logits.permute(0, 2, 1), label, reduction="mean")
            # NLL_loss is setting as reduction = "none"

            total_elbo += loss.item() * batch_size
            total_ppl += ppl.item() * batch_size
            total_samples += batch_size

        return total_elbo/total_samples, np.exp(total_ppl/total_samples)
