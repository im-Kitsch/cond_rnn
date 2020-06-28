import torch
from util.traj_dataset_general import TrajDatasetGeneral, wrap_data

class VanillaVAEModel(torch.nn.Module):
    def __init__(self, n_word, embed_size, gru_hid_size, latent_dim):
        super(VanillaVAEModel, self).__init__()
        self.emb = torch.nn.Embedding(n_word, embed_size)
        self.enc_gru = torch.nn.GRU(embed_size, gru_hid_size, num_layers=2, dropout=0.2)
        self.enc_mu = torch.nn.Linear(gru_hid_size, latent_dim)
        self.enc_log_var = torch.nn.Linear(gru_hid_size, latent_dim)
        self.dec_nn = torch.nn.Linear(latent_dim, gru_hid_size)
        self.dec_gru = torch.nn.GRU(gru_hid_size, embed_size)
        self.dec_nn = torch.nn.Linear(embed_size, n_word)

        return

    def forward(self, input_data):
        seq_len = input_data.shape[0]
        latent_z, mu, log_var = self.enc_forward(input_data)
        latent_z = latent_z.unsqueeze(0)
        latent_z = latent_z.repeat(seq_len, 1, 1)
        latent_z = self.dec_nn(latent_z)
        logits, _ = self.dec_gru(latent_z)
        logits = self.dec_nn(logits)
        logits = torch.log_softmax(logits, dim=2)
        return logits, mu, log_var

    def enc_forward(self, input_data, h0=None):
        outputs = self.emb(input_data)
        outputs = self.enc_gru(outputs)
        outputs = outputs[-1]
        outputs_mu = self.enc_mu(outputs)
        outputs_log_var = self.enc_log_var(outputs)
        outputs_sigma = torch.exp(0.5 * outputs_log_var)
        outputs = torch.normal(0, 1, outputs_mu.shape)
        outputs = outputs_mu + outputs * outputs_sigma
        return outputs, outputs_mu, outputs_log_var

    def dec_forward(self):
        return


class VanillaVAE:
    def __init__(self, n_word, embed_size, gru_hid_size, latent_dim, lr):
        self.vae_model = VanillaVAEModel(n_word, embed_size, gru_hid_size, latent_dim)
        self.optimizer = torch.optim.Adam(self.vae_model.parameters(), lr=lr)
        self.criterion = torch.nn.NLLLoss()
        return

    def fit(self, training_loader, validation_loader, epochs, device, optimizer, criterion):
        for i, batch_data in training_loader:
            batch_data = batch_data.T.to(device)
            logits, mu, log_var = self.vae_model.forward(batch_data)
            loss_cre = self.criterion(logits.perpute(0, 2, 1), batch_data)
            loss_rec = -0.5 *  (log_var - mu**2 - torch.exp(log_var))
            loss_rec = loss_rec.mean()
            loss = loss_cre + loss_rec

            self.vae_model.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(i, loss.item())
        return





