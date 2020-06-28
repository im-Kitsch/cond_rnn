import torch


class TestDT(torch.utils.data.Dataset):
    def __init__(self, src):
        super(TestDT, self).__init__()

        self.src = src
        self.keys = src.keys
        return

    def __len__(self):
        return self.src[0].shape[0]

    def __getitem__(self, idx):
        return {_key:_item[idx] for _key, _item in self.src.items()}

dt = {"a":torch.rand(16, 3), "b":torch.rand(16, 1)}

training_loader = torch.utils.data.DataLoader(
                    dt,
                    batch_size=5,
                    shuffle=True, num_workers=4,
                    drop_last=False)

it_dt = iter(training_loader)
next(it_dt)