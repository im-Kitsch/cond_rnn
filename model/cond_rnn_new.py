import torch
import torchsummaryX
import numpy as np
from model.base_rnn import BaseRNN


class CondGRU(BaseRNN):
    def __init__(self, num_class, hid_size, hid_layers, drop_out_rate,
                 embedding_size, cond_range, emb_cond_size,
                 learning_rate, data_keys, writer_comment="cond_rnn"):
        super(CondGRU, self).__init__(data_keys=data_keys, writer_comment=writer_comment)

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

        cond1 = self.cond_emb1(condition[[0]])
        cond2 = self.cond_emb2(condition[[1]])
        cond3 = self.cond_emb3(condition[[2]])
        cond = torch.cat([cond1, cond2, cond3], dim=2)
        cond = cond.repeat(seq_len, 1, 1)
        embeded_data = self.embedding(input_data)
        concat_data = torch.cat([cond, embeded_data], dim=2)
        output, hn = self.gru(concat_data, h0)
        output = self.dense(output)
        output = self.log_softmax(output)
        return output, hn

    def forward_packed_data(self, batch_data, h0=None, teaching_ratio=0.):
        if teaching_ratio == 0.:
            input_data = batch_data["data"]
            cond = batch_data["cond"]
            pred_logits, hn = self.forward(input_data, cond, h0)
            return pred_logits, hn
        else:
            raise Exception("not implemented")







