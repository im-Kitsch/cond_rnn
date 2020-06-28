import torch
import torchsummaryX
import numpy as np


class AttentionBahdanau(torch.nn.Module):
    def __init__(self, hid_source, hid_target):
        super(AttentionBahdanau, self).__init__()
        self.attn = torch.nn.Linear(hid_source+hid_target, hid_target)
        self.v = torch.nn.Linear(hid_target, 1, bias=False)
        return

    # [1,B,H_de]， [T_en,B,H_en] -> [1,B,H_en] [T_en, B]
    def forward(self, hid_decoder, hid_encoder):
        hid_encoder = hid_encoder.transpose(0, 1)
        T_en = hid_encoder.shape[1]
        hid_decoder = hid_decoder.transpose(0, 1)
        hid_decoder = hid_decoder.repeat(1, T_en, 1)

        weights = torch.cat([hid_encoder, hid_decoder], dim=2)
        weights = torch.tanh(self.attn(weights))
        weights = self.v(weights)
        weights = weights.squeeze(dim=2)
        weights = torch.softmax(weights, dim=1)
        weights = weights.unsqueeze(dim=1)
        c_i = torch.bmm(weights, hid_encoder)
        return c_i.transpose(0,1), weights.squeeze(1).T


class CondGruAttentionSeq(torch.nn.Module):
    def __init__(self, num_class, hid_size, num_cond_seq_class,
                 embedding_size, cond_range, emb_cond_size,
                 learning_rate):
        super(CondGruAttentionSeq, self).__init__()

        self.emb_cond = torch.nn.ModuleList(
                            [torch.nn.Embedding(_emb, emb_cond_size)
                             for _emb in cond_range] )
        self.attention = AttentionBahdanau(hid_target=hid_size,
                                           hid_source=emb_cond_size)

        self.embedding = torch.nn.Embedding(num_class, embedding_size)
        self.gru = torch.nn.GRU(embedding_size, hid_size)

        self.dense = torch.nn.Linear(emb_cond_size+embedding_size+hid_size, num_class)

        self.num_class = num_class
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(reduction="mean")

        self.cond_seq_emb = torch.nn.Embedding(num_cond_seq_class, emb_cond_size)
        self.gru_cond = torch.nn.GRU(emb_cond_size, emb_cond_size)
        return

    # input_data [Seq, B], condition [:, B]
    def forward(self, input_data, condition, cond_seq, h0=None):
        assert input_data.ndim == 2
        assert condition.ndim == 2
        assert condition.shape[1] == input_data.shape[1]

        if h0 is None:
            h0 = self._init_hidden(batch_size=input_data.shape[1],
                                   device=input_data.device)
        cond_emb = self.cond_embedding(condition)

        cond_emb_seq, _ = self.gru_cond(self.cond_seq_emb(cond_seq))
        cond_emb = torch.cat([cond_emb, cond_emb_seq], dim=0)

        total_pred = []
        weights = []
        for _i in range(input_data.shape[0]):
            input_word = input_data[[_i], :]
            input_emb = self.embedding(input_word)
            c_i, w_i = self.attention(hid_decoder=h0, hid_encoder=cond_emb)

            out, hn = self.gru(input_emb, h0)
            pred = torch.cat([c_i, input_emb, h0], dim=2) #TODO hn or h0
            pred = self.dense(pred)
            pred = torch.log_softmax(pred, dim=2)

            h0 = hn#.detach()
            total_pred.append(pred)
            w_i = w_i.T
            weights.append(w_i.unsqueeze(0))

        total_pred = torch.cat(total_pred, dim=0)
        weights = torch.cat(weights, dim=0)
        return total_pred, hn, weights

    def cond_embedding(self, cond):
        cond_num = cond.shape[0]
        cond_emb = torch.cat(
            [self.emb_cond[_i](cond[[_i], :])
             for _i in range(cond_num)], dim=0)
        return cond_emb

    def _init_hidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.gru.hidden_size, device=device)

    def train_step(self, input_data, condition, label, cond_seq):
        pred_logits, _, _ = self.forward(input_data=input_data,
                                         condition=condition,
                                         cond_seq=cond_seq)
        pred_logits = pred_logits.reshape(-1, self.num_class)
        label = label.flatten()

        self.optimizer.zero_grad()
        loss = self.criterion(pred_logits, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # TODO respectively mean of sentence and sample
    def eval_perplexity(self, data_loader, device):
        total_loss = 0.
        num_sample = 0.

        for i_bt, (batch_data, batch_label, batch_cond, batch_cond_seq) in enumerate(data_loader):
            batch_data, batch_label, batch_cond = batch_data.to(device), batch_label.to(device), batch_cond.to(device)
            batch_data, batch_label, batch_cond = batch_data.T, batch_label.T, batch_cond.T
            batch_cond_seq = batch_cond_seq.to(device).T
            logits, _, _ = self.forward(batch_data, batch_cond, batch_cond_seq)

            logits = logits.reshape(-1, self.num_class)
            batch_label = batch_label.flatten()
            loss = self.criterion(logits, batch_label)
            total_loss += loss.item() * batch_cond.shape[1]
            num_sample += batch_cond.shape[1]

        return np.exp(total_loss/num_sample)

if __name__ == "__main__":
    cond_gru_attn = CondGruAttention(500, 15, 30, [4, 5, 11], 7, 1e-5)
    input_dt = torch.randint(0, 500, (7, 8))
    cond = torch.randint(0, 4, (3, 8))
    cond_gru_attn(input_dt, cond, batch_first=False)
