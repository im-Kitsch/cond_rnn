import torch
import torchsummaryX
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from deprecated.data_loader import TrajDatasetCondSeq, load_condition, load_pflow, load_cond_seq

from model.cond_rnn_attention_seq import CondGruAttentionSeq


def show_attention(ax, fig, attention, input_condition, sentence):

    cax = ax.matshow(attention, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([" "] + input_condition, rotation=90)
    ax.set_yticklabels([" "] + sentence)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    return ax

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--begin-index', type=int, default=6*60, metavar='N', help='--begin-index')
parser.add_argument('--time-interval', type=int, default=30, metavar='N', help='--time-interval')
parser.add_argument('--num-per-day', type=int, default=18*2, metavar='N', help='--num-per-day')
parser.add_argument('--batch-size', type=int, default=512, metavar='N', help='--batch-size')

parser.add_argument('--embed-size', type=int, default=120, metavar='N', help='--embed-size')
parser.add_argument('--hid-size', type=int, default=80, metavar='N', help='hidden size')
parser.add_argument('--cond_emb-size', type=int, default=10, metavar='N')

parser.add_argument('--epochs', type=int, default=35, metavar='N')
parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate')

args, unknown = parser.parse_known_args()

torch.manual_seed(3)
np.random.seed(3)
DICT_FOLDER = "./dict_file"
PFLOW_DATA_PATH = "~/data/pflow_mini_preprocessed.csv"
COND_PATH = "~/data/pflow_mini_condition.csv"

DEVICE = torch.device("cuda:0")

cond = load_condition(COND_PATH)
COND_RANGE = (cond.max(axis=0)+1).tolist()

dataset, pflow_translator = load_pflow(PFLOW_DATA_PATH, DICT_FOLDER,
                                       begin_ix=args.begin_index,
                                       time_interval=args.time_interval,
                                       num_per_day=args.num_per_day,
                                       begin_code=None)
cond_seq = load_cond_seq(PFLOW_DATA_PATH,
                         begin_ix=args.begin_index,
                         time_interval=args.time_interval * 2,
                         num_per_day=args.num_per_day//2)

NUM_CLASS = pflow_translator.num_class

num_traning_sample = int(dataset.shape[0] * 0.8)
training_data, training_label, training_cond = dataset[:num_traning_sample, :-1], dataset[:num_traning_sample, 1:], cond[:num_traning_sample]
validation_data, validation_label, validation_cond = dataset[num_traning_sample:, :-1], dataset[num_traning_sample:, 1:], cond[num_traning_sample:]
training_cond_seq, validation_cond_seq = cond_seq[:num_traning_sample], cond_seq[num_traning_sample:]

training_loader = torch.utils.data.DataLoader(
            TrajDatasetCondSeq(training_data, training_label, training_cond, training_cond_seq),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)
validation_loader = torch.utils.data.DataLoader(
            TrajDatasetCondSeq(validation_data, validation_label, validation_cond, validation_cond_seq),
            batch_size=args.batch_size,
            shuffle=True, num_workers=4,
            drop_last=False)

cond_attn = CondGruAttentionSeq(num_class=NUM_CLASS, hid_size=args.hid_size, num_cond_seq_class=cond_seq.max()+1,
                        embedding_size=args.embed_size, cond_range=COND_RANGE,
                        emb_cond_size=args.cond_emb_size, learning_rate=args.lr)

# test model
test_dt, test_cond = training_loader.dataset[0:10][0].T, training_loader.dataset[0:10][2].T
test_dt, test_cond = torch.tensor(test_dt), torch.tensor(test_cond)
test_cond_seq = torch.tensor(training_loader.dataset[0:10][3].T)
model_summary = torchsummaryX.summary(cond_attn, test_dt, test_cond, test_cond_seq)

cond_attn.to(DEVICE)

for _epoc in range(args.epochs):
    cond_attn.train()
    with tqdm(total=len(training_loader), desc=f"{_epoc:2.0f}") as pbar:
        for i_bt, (batch_data, batch_label, batch_cond, batch_cond_seq) in enumerate(training_loader):

            loss = cond_attn.train_step(input_data=batch_data.to(DEVICE).T,
                                        label=batch_label.to(DEVICE).T,
                                        condition=batch_cond.to(DEVICE).T,
                                        cond_seq=batch_cond_seq.to(DEVICE).T)

            pbar.set_postfix({"loss": f"{loss:.3f}"})
            pbar.update()

        with torch.no_grad():
            cond_attn.eval()
            training_ppl = cond_attn.eval_perplexity(training_loader, device=DEVICE)
            validation_ppl = cond_attn.eval_perplexity(validation_loader, device=DEVICE)

        pbar.set_postfix({"ppl": f"{training_ppl:.1f}", "val_ppl":f"{validation_ppl:.1f}"})

    if (_epoc+1)%5 == 0:
        cond_attn.to("cpu")
        with torch.no_grad():
            batch_data, batch_label, batch_cond, batch_cond_seq = training_loader.dataset[[0]]
            batch_data, batch_label, batch_cond = torch.tensor(batch_data).T, torch.tensor(batch_label).T, torch.tensor(
                batch_cond).T
            batch_cond_seq = torch.tensor(batch_cond_seq).T

            _, _, weights = cond_attn(batch_data, batch_cond, batch_cond_seq)
            weights = weights.squeeze(1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax = show_attention(ax, fig=fig, attention=weights.numpy(),
                                sentence=batch_data.flatten().tolist(),
                                input_condition=["6", "7", "9"])
        cond_attn.to(DEVICE)
