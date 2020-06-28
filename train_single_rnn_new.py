#%%
import torchsummaryX
import torch
import argparse
from tqdm import tqdm

import numpy as np

from model.single_rnn_new import SingleRNN
from util.traj_dataset_general import TrajDatasetGeneral, wrap_data
from util.data_prepare import traj_prepare


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--begin-index', type=int, default=6*60, metavar='N', help='--begin-index')
parser.add_argument('--time-interval', type=int, default=15, metavar='N', help='--time-interval')
parser.add_argument('--num-per-day', type=int, default=18*4, metavar='N', help='--num-per-day')
parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='--batch-size')
parser.add_argument('--embed-size', type=int, default=40, metavar='N', help='--embed-size')
parser.add_argument('--epochs', type=int, default=10, metavar='N')
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

num_samples = 50000
traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                        num_samples,
                                        use_cols=args.begin_index+np.arange(args.num_per_day)*args.time_interval)
                                        #add_idx=["0/SOS"])

NUM_CLASS = n_traj
DATA_KEYS = ["data", "label"]

num_training_sample = int(num_samples * 0.8)
training_data, validation_data= traj[:num_training_sample, :-1], traj[num_training_sample:, :-1]
training_label, validation_label= traj[:num_training_sample, 1:], traj[num_training_sample:, 1:]

training_loader = torch.utils.data.DataLoader(
            TrajDatasetGeneral([training_data, training_label]),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)
validation_loader = torch.utils.data.DataLoader(
            TrajDatasetGeneral([validation_data, validation_label]),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)

DEVICE = torch.device("cuda:0")


sing_gru = SingleRNN(num_class=NUM_CLASS,
                     hid_size=args.hid_size, hid_layers=args.hid_layer,
                     drop_out_rate=args.drop_out, embedding_size=args.embed_size,
                     learning_rate=args.lr,
                     data_keys=DATA_KEYS,
                     writer_comment="Single_RNN")


test_dt = training_loader.dataset[0:10][0].T
test_dt = torch.tensor(test_dt)
model_summary = torchsummaryX.summary(sing_gru, test_dt)
#%%
sing_gru.to(DEVICE)

sing_gru.fit(training_loader=training_loader, validation_loader=validation_loader,
             epochs=args.epochs, device=DEVICE,
             optimizer=sing_gru.optimizer, criterion=sing_gru.criterion)
log_dir = sing_gru.writer.log_dir
sing_gru.writer.close()
torch.save(sing_gru, f"{log_dir}/model.pt")
#%%

data = training_loader.dataset.get_sample(15)
begin_time = int(args.num_per_day * 0.35)
data[0] = data[0][:, :begin_time]
data = wrap_data(data, keys=DATA_KEYS, device=DEVICE)
sampled_traj = sing_gru.sample_traj(data, predict_length=args.num_per_day-args.begin_index,
                                    sample_method="multinomial")
