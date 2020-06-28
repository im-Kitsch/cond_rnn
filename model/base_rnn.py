import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from util.traj_dataset_general import wrap_data
from util.nlp_util import multinomial_choose


class BaseRNN(torch.nn.Module):
    def __init__(self, data_keys, writer_comment):
        super(BaseRNN, self).__init__()
        self.data_keys = data_keys
        self.writer = SummaryWriter(comment=writer_comment)
        return

    def forward(self, *input, **kwargs):
        raise NotImplementedError

    def forward_packed_data(self, batch_data, h0=None, teaching_ratio=0.):

        raise NotImplementedError

    def fit(self, training_loader, validation_loader, epochs, device, optimizer, criterion, teach_forcing=0.):
        # TODO not proper transfer of optimizer and criterion
        # TODO teach forcing ratio
        total_step = 0
        for _epoc in range(epochs):
            total_loss = 0.
            num_sample = 0.
            with tqdm(total=len(training_loader), desc=f"{_epoc:2.0f}") as pbar:
                for i_bt, batch_data in enumerate(training_loader):
                    batch_data = wrap_data(batch_data, self.data_keys, device=device)
                    batch_size = batch_data["data"].shape[1]

                    loss = self.train_step(batch_data, optimizer=optimizer, criterion=criterion)

                    pbar.set_postfix({"loss": f"{loss:.3f}"})
                    pbar.update()

                    total_loss += loss
                    num_sample += batch_size

                    total_step += 1
                    self.writer.add_scalar("training_loss", loss, total_step)

                training_ppl = self.eval_ppl(training_loader, criterion=criterion, device=device)
                validation_ppl = self.eval_ppl(validation_loader, criterion=criterion, device=device)
                pbar.set_postfix({"ppl": f"{training_ppl:.1f}", "val_ppl": f"{validation_ppl:.1f}"})
                self.writer.add_scalar("ppl/train", training_ppl, _epoc)
                self.writer.add_scalar("ppl/val", validation_ppl, _epoc)
        return

    @torch.no_grad()
    def eval_ppl(self, data_loader, criterion, device=None):
        total_loss = 0.
        num_sample = 0.

        for i_bt, batch_data in enumerate(data_loader):
            batch_data = wrap_data(batch_data, self.data_keys, device=device)
            batch_size = batch_data["data"].shape[1]

            output, hn = self.forward_packed_data(batch_data, h0=None, teaching_ratio=0.)
            label = batch_data["label"]
            output = output.permute(0, 2, 1)
            loss = criterion(output, label)

            total_loss += loss.item() * batch_size
            num_sample += batch_size

        return np.exp(total_loss/num_sample)

    def beam_search(self, beam_size):
        return

    @torch.no_grad()
    def sample_traj(self, given_data, predict_length, sample_method="greedy", topk=3):
        # sample_method: multinomial or greedy
        assert given_data["data"].shape[1] == 1 # batch_size is 1
        ht = None
        data_t = given_data
        pred_traj = []

        for _t in range(predict_length):
            logits, ht = self.forward_packed_data(data_t, ht, teaching_ratio=0.)
            logits = logits[[-1]]
            if sample_method == "multinomial":
                _, pred_item = multinomial_choose(logits=logits, topk=topk)
            elif sample_method == "greedy":
                _, pred_item = torch.max(logits, dim=2)
            else:
                raise Exception("not implemented sample method")
            pred_traj.append(pred_item.item())
            given_data["data"] = pred_item
        return np.array(pred_traj)

    def train_step(self, batch_data, optimizer, criterion):
        output, hn = self.forward_packed_data(batch_data)
        output = output.permute(0, 2, 1)
        label = batch_data["label"]

        optimizer.zero_grad()
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        return loss.item()
