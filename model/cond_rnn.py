import torch
import torchsummaryX
import numpy as np


class CondGRU(torch.nn.Module):
    def __init__(self, num_class, hid_size, hid_layers, drop_out_rate,
                 embedding_size, cond_range, emb_cond_size,
                 learning_rate):
        super(CondGRU, self).__init__()

        emb_cond_size1, emb_cond_size2, emb_cond_size3 = emb_cond_size
        cond_range1, cond_range2, cond_range3 = cond_range
        self.cond_emb1 = torch.nn.Embedding(cond_range1, emb_cond_size1)
        self.cond_emb2 = torch.nn.Embedding(cond_range2, emb_cond_size2)
        self.cond_emb3 = torch.nn.Embedding(cond_range3, emb_cond_size3)

        self.embedding = torch.nn.Embedding(num_class, embedding_size)
        self.gru = torch.nn.GRU(
            input_size=embedding_size + emb_cond_size1 + emb_cond_size2 + emb_cond_size3,
            hidden_size=hid_size,
            num_layers=hid_layers, dropout=drop_out_rate,
            bidirectional=False, bias=True,
        )
        self.dense = torch.nn.Linear(hid_size*1, num_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

        self.num_class = num_class

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.99)
        self.criterion = torch.nn.NLLLoss(reduction="mean")
        return

    def forward(self, input_data, condition, h0=None):
        seq_len = input_data.shape[0]
        condition = condition.unsqueeze(0)
        condition = condition.repeat(seq_len, 1, 1)
        cond1 = self.cond_emb1(condition[:, :, 0])
        cond2 = self.cond_emb2(condition[:, :, 1])
        cond3 = self.cond_emb3(condition[:, :, 2])
        cond = torch.cat([cond1, cond2, cond3], dim=2)
        embeded_data = self.embedding(input_data)
        concat_data = torch.cat([cond, embeded_data], dim=2)
        output, hn = self.gru(concat_data, h0)
        output = self.dense(output)
        output = self.log_softmax(output)
        return output, hn

    def train_step(self, input_data, condition, label):
        output, hn = self.forward(input_data, condition)
        output = output.reshape(-1, self.num_class)
        label = label.flatten()

        self.optimizer.zero_grad()
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_perplexity(self, data_loader, device):
        total_loss = 0.
        num_sample = 0.

        for i_bt, (batch_data, batch_label, batch_cond) in enumerate(data_loader):
            batch_data, batch_label, batch_cond = batch_data.to(device).T, batch_label.to(device).T, batch_cond.to(device)
            logits, _ = self.forward(batch_data, batch_cond)
            logits = logits.reshape(-1, self.num_class)
            batch_label = batch_label.flatten()
            loss = self.criterion(logits, batch_label)
            total_loss += loss.item() * batch_cond.shape[0]
            num_sample += batch_cond.shape[0]

        return np.exp(total_loss/num_sample)


