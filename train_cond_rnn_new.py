import torchsummaryX
import torch
import argparse
from tqdm import tqdm

import numpy as np

from model.cond_rnn_new import CondGRU
from util.traj_dataset_general import TrajDatasetGeneral

from util.data_prepare import cond_prepare, traj_prepare

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--begin-index', type=int, default=6*60, metavar='N', help='--begin-index')
parser.add_argument('--time-interval', type=int, default=15, metavar='N', help='--time-interval')
parser.add_argument('--num-per-day', type=int, default=18*4, metavar='N', help='--num-per-day')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='--batch-size')
parser.add_argument('--embed-size', type=int, default=40, metavar='N', help='--embed-size')
parser.add_argument('--cond_embed', nargs='+', type=int, default=[4, 10, 10])
parser.add_argument('--epochs', type=int, default=50, metavar='N')
parser.add_argument('--hid-size', type=int, default=80, metavar='N', help='hidden size')
parser.add_argument('--hid-layer', type=int, default=2, metavar='N', help='hidden layer')
parser.add_argument('--drop-out', type=float, default=0.2, metavar='DR_R', help='drop out rate')
parser.add_argument('--lr', type=float, default=5e-3, metavar='LR', help='learning rate')

args, unknown = parser.parse_known_args()
torch.manual_seed(3)
np.random.seed(3)

source_folder = "/home/ubuntu/pflow_data/filtered"
traj_path = f"{source_folder}/traj.csv"
code_path = f"{source_folder}/code.csv"
cond_path = f"{source_folder}/cond.csv"

num_samples = 5000

traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                        num_samples,
                                        use_cols=args.begin_index+np.arange(args.nums_per_day)*args.time_interval)
                                        #, add_idx=["0/SOS"])
#TODO check here
cond, n_categ = cond_prepare(cond_path, num_samples)

NUM_CLASS = n_traj

num_training_sample = int(num_samples * 0.8)
training_data, validation_data= traj[:num_training_sample, :-1], traj[num_training_sample:, :-1]
training_label, validation_label= traj[:num_training_sample, 1:], traj[num_training_sample:, 1:]
training_cond, validation_cond= cond[:num_training_sample], cond[num_training_sample:]

training_loader = torch.utils.data.DataLoader(
            TrajDatasetGeneral([training_data, training_label, training_cond]),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)
validation_loader = torch.utils.data.DataLoader(
            TrajDatasetGeneral([validation_data, validation_label, validation_cond]),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)

DATA_KEYS = ["data", "label", "cond"]

DEVICE = torch.device("cuda:0")
#%%
cond_gru = CondGRU(NUM_CLASS, hid_size=args.hid_size, hid_layers=args.hid_layer,
                   drop_out_rate=args.drop_out,
                   embedding_size=args.embed_size, cond_range=n_categ,
                   emb_cond_size=args.cond_embed, learning_rate=args.lr,
                   data_keys=DATA_KEYS)

test_dt, test_cond = training_loader.dataset[0:10][0].T, training_loader.dataset[0:10][2].T
test_dt, test_cond = torch.tensor(test_dt), torch.tensor(test_cond)
model_summary = torchsummaryX.summary(cond_gru, test_dt, test_cond)

cond_gru.to(DEVICE)

cond_gru.fit(training_loader, validation_loader, epochs=args.epochs, device=DEVICE,
             optimizer=cond_gru.optimizer,
             criterion=cond_gru.criterion)


