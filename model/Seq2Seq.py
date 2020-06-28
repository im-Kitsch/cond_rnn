import torch

from model.base_rnn import BaseEncoderDecoder

class Seq2Seq(BaseEncoderDecoder):
    def __init__(self, emb_dim, n_traj_code, hid_dim, n_layer, drop_rate):
        self.emb = torch.nn.Embedding(n_traj_code, emb_dim)
        self.enc = torch.nn.GRU(emb_dim, hid_dim,
                                num_layers=n_layer, dropout=drop_rate)
        self.dec = torch.nn.GRU(emb_dim, hid_dim,
                                num_layers=n_layer, dropout=drop_rate)
        self.nn = torch.nn.Linear(hid_dim, n_traj_code)
        return

    def batch_data_forward(self, batch_data):

        return

    def eval_perplexity(self, data_loader):

        return

    def sample_traj(self, data):

        return
