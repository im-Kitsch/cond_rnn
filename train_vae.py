import torchsummaryX
import torch
import argparse
from tqdm import tqdm

import numpy as np

from model.seq_vae import SeqVAE
from util.traj_dataset_general import TrajDatasetGeneral, wrap_data
from util.data_prepare import traj_prepare
from util.plot_util import visualization_trajectory


def train(args):
    traj, n_traj, translator = traj_prepare(
                                TRAJ_PATH, CODE_PATH, NUM_SAMPLES,
                                use_cols=args.begin_index + np.arange(args.num_per_day) * args.time_interval,
                                add_idx=ADD_IDX)
    NUM_CLASS = n_traj

    num_training_sample = int(NUM_SAMPLES * 0.8)
    ix_sos = translator.trans_code2ix(["0/SOS"])[0]  # list to item
    traj = np.concatenate([np.ones((traj.shape[0], 1), dtype=int),
                           traj], axis=1)
    training_data, validation_data = traj[:num_training_sample, :-1], traj[num_training_sample:, :-1]
    training_label, validation_label = traj[:num_training_sample, 1:], traj[num_training_sample:, 1:]

    training_loader = torch.utils.data.DataLoader(
        TrajDatasetGeneral([training_data, training_label]),
        batch_size=args.batch_size,
        shuffle=True, num_workers=4,
        drop_last=False)
    validation_loader = torch.utils.data.DataLoader(
        TrajDatasetGeneral([validation_data, validation_label]),
        batch_size=2000,
        shuffle=False, num_workers=4,
        drop_last=False)

    seq_vae = SeqVAE(num_class=NUM_CLASS, embedding_size=args.embed_size, embedding_drop_out=args.emb_dropout,
                     hid_size=args.hid_size, hid_layers=args.hid_layer, latent_z_size=args.latent_z_size,
                     word_dropout_rate=args.word_dropout, gru_drop_out=args.gru_dropout,
                     anneal_k=args.anneal_k, anneal_x0=args.anneal_x0, anneal_function=args.anneal_func,
                     data_keys=DATA_KEYS, learning_rate=args.lr,
                     sos_idx=ix_sos, unk_idx=None)

    test_batch = wrap_data(next(iter(training_loader)), DATA_KEYS)
    model_summary = torchsummaryX.summary(seq_vae.model, test_batch['data'])

    seq_vae.model.to(DEVICE)

    seq_vae.fit(training_loader=training_loader, validation_loader=validation_loader, epochs=args.epochs, device=DEVICE)
    return seq_vae, translator


def vis_eval(seq_vae, translator):
    z = seq_vae.model.sample_z(15, device=DEVICE)
    pred_traj = seq_vae.model.inference(z=z, pred_length=72, batch_first_z=True, device=DEVICE)

    pred_traj = pred_traj.T
    pred_traj = pred_traj[:, 1:]  # drop SOS
    pred_traj = pred_traj.cpu().numpy()
    pred_traj = translator.trans_ix2coordinate(pred_traj)

    for i in range(pred_traj.shape[0]):

        _traj = pred_traj[[i]]  # keep dim
        fig1, fig2 = visualization_trajectory(_traj, 1, 1, plot_3d=True)
        seq_vae.writer.add_figure("sample_2d", fig1, i)
        seq_vae.writer.add_figure("sample_3d", fig2, i)
    return pred_traj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--begin-index', type=int, default=6*60, metavar='N', help='--begin-index')
    parser.add_argument('--time-interval', type=int, default=15, metavar='N', help='--time-interval')
    parser.add_argument('--num-per-day', type=int, default=18*4, metavar='N', help='--num-per-day')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='--batch-size')
    parser.add_argument('--embed-size', type=int, default=40, metavar='N', help='--embed-size')
    parser.add_argument('--emb-dropout', type=float, default=0., metavar='EmbDrop')

    parser.add_argument('--latent-z-size', type=int, default=15, metavar='N')
    parser.add_argument('--word-dropout', type=float, default=0., metavar='WDrop')

    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--hid-size', type=int, default=80, metavar='N', help='hidden size')
    parser.add_argument('--hid-layer', type=int, default=2, metavar='N', help='hidden layer')
    parser.add_argument('--gru-dropout', type=float, default=0.2, metavar='DR_R', help='drop out rate')
    parser.add_argument('--lr', "--learning_rate", type=float, default=5e-3, metavar='LR', help='learning rate')

    parser.add_argument('--anneal-k', type=float, default=0.0025, metavar='AnK')
    parser.add_argument('--anneal-x0', type=float, default=2500, metavar='AnX0')
    parser.add_argument('--anneal-func', type=str, default='logistic')

    args, unknown = parser.parse_known_args()
    torch.manual_seed(3)
    np.random.seed(3)

    SOURCE_FOLDER = "/home/ubuntu/pflow_data/filtered"
    TRAJ_PATH = f"{SOURCE_FOLDER}/traj.csv"
    CODE_PATH = f"{SOURCE_FOLDER}/code.csv"

    NUM_SAMPLES = 30000
    ADD_IDX =["0/SOS"]
    DATA_KEYS = ["data", "label"]

    DEVICE = torch.device("cuda:0")

    trained_model, translator = train(args)
    vis_eval(trained_model, translator)
