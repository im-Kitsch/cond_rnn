import torch

from model.base_rnn import BaseRNN

class SingleRNN(BaseRNN):
    def __init__(self, num_class, embedding_size, hid_size, hid_layers, drop_out_rate,
                 learning_rate, data_keys, writer_comment):
        super(SingleRNN, self).__init__(data_keys, writer_comment)
        self.embedding = torch.nn.Embedding(num_class, embedding_size)
        self.gru = torch.nn.GRU(
            input_size=embedding_size,
            hidden_size=hid_size,
            num_layers=hid_layers, dropout=drop_out_rate,
            bias=True,
            bidirectional=False
        )
        self.dense = torch.nn.Linear(hid_size, num_class)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

        self.num_class = num_class

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(reduction="mean")
        return

    def forward(self, input_data, h0=None):
        embeded = self.embedding(input_data)
        logits, hn = self.gru(embeded, h0)
        logits = self.dense(logits)
        logits = self.log_softmax(logits)
        return logits, hn

    def forward_packed_data(self, batch_data, h0=None, teaching_ratio=0.):
        if teaching_ratio == 0.:
            input_data = batch_data["data"]
            pred_logits, hn = self.forward(input_data, h0)
            return pred_logits, hn
        else:
            raise Exception("not implemented")


