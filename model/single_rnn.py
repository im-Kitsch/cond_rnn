import torch
import torchsummaryX
import numpy as np


class SingleGRU(torch.nn.Module):
    def __init__(self, num_class, hid_size, hid_layers, drop_out_rate,
                 embedding_size, learning_rate):
        super(SingleGRU, self).__init__()

        self.embedding = torch.nn.Embedding(num_class, embedding_size)
        self.gru = torch.nn.GRU(
            input_size=embedding_size,
            hidden_size=hid_size,
            num_layers=hid_layers, dropout=drop_out_rate,
            bidirectional=False, bias=True
        )
        self.dense = torch.nn.Linear(hid_size*1, num_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

        self.num_class = num_class

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(reduction="mean")
        return

    def forward(self, input_data, h0=None):
        seq_len = input_data.shape[0]

        embeded_data = self.embedding(input_data)

        output, hn = self.gru(embeded_data, h0)
        output = self.dense(output)
        output = self.log_softmax(output)
        return output, hn

    def train_step(self, input_data, label):
        output, hn = self.forward(input_data)
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

        for i_bt, (batch_data, batch_label, _) in enumerate(data_loader):
            batch_data, batch_label = batch_data.to(device).T, batch_label.to(device).T
            logits, _ = self.forward(batch_data)
            logits = logits.reshape(-1, self.num_class)
            batch_label = batch_label.flatten()
            loss = self.criterion(logits, batch_label)
            total_loss += loss.item() * batch_data.shape[1]
            num_sample += batch_data.shape[1]

        return np.exp(total_loss/num_sample)


