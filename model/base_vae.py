import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from util.traj_dataset_general import wrap_data


class Annealer:
    def __init__(self, k, x0, anneal_function):
        self.k = k
        self.x0 = x0
        self._step_count = 0.
        self.anneal_function = anneal_function
        return

    def step(self, count=True):
        if self.anneal_function == 'logistic':
            weight = float(1 / (1 + np.exp(-self.k * (self._step_count - self.x0))))
        elif self.anneal_function == 'linear':
            weight = min(1, self._step_count / self.x0)
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


class BaseVAE:
    def __init__(self, model, anneal_k, anneal_x0, anneal_function, data_keys,
                 learning_rate, writer_comment):
        self.data_keys = data_keys
        self.model = model
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
        loss_kl = loss_kl.sum(dim=0) # TODO  dim 0 or dim 1?
        return loss_nll.mean(), loss_kl.mean()

    def fit(self, training_loader, validation_loader, epochs, device):
        total_step = 0
        for _epoc in range(epochs):
            with tqdm(total=len(training_loader), desc=f"{_epoc:2.0f}", ncols=100) as pbar:
                self.model.train()
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

                print(
                    f"\r\n evaluation: ppl: {training_ppl:.1f}, val_ppl: {validation_ppl:.1f}, elbo: {training_elbo:.1f} val elbo: {validation_elbo:.1f}")
                # pbar.set_postfix({"ppl": f"{training_ppl:.1f}", "val_ppl": f"{validation_ppl:.1f}",
                #                   "elbo": f"{training_elbo:.1f}", "val_elbo": f"{validation_elbo:.1f}"})
                self.writer.add_scalar("ppl/train", training_ppl, _epoc)
                self.writer.add_scalar("ppl/val", validation_ppl, _epoc)
                self.writer.add_scalar("elbo/train", training_elbo, _epoc)
                self.writer.add_scalar("elbo/val", validation_elbo, _epoc)
        return

    def train_step(self, batch_data):
        label = batch_data["label"]
        logits, mean, logvar = self.model.batch_forward(batch_data)
        loss_nll, loss_kl = self.loss(logits, label, mean, logvar)
        weight = self.annealer.step()
        loss = loss_nll + loss_kl * weight

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_nll.item(), loss_kl.item(), weight

    @torch.no_grad()
    def eval_metric(self, data_loader, device):
        self.model.eval()

        total_ppl = 0
        total_elbo = 0
        total_samples = 0
        weight = self.annealer.weight
        for i_bt, batch_data in enumerate(data_loader):
            batch_data = wrap_data(batch_data, self.data_keys, device=device)
            label = batch_data["label"]
            batch_size = label.shape[1]

            logits, mean, logvar = self.model.batch_forward(batch_data)
            loss_nll, loss_kl = self.loss(logits, label, mean, logvar)

            loss = loss_nll + loss_kl * weight

            ppl = torch.nn.functional.nll_loss(logits.permute(0, 2, 1), label, reduction="mean")
            # NLL_loss is setting as reduction = "none"

            total_elbo += loss.item() * batch_size
            total_ppl += ppl.item() * batch_size
            total_samples += batch_size

        return total_elbo / total_samples, np.exp(total_ppl / total_samples)
