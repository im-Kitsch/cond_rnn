import torch

#TODO 重写 不好的表达, 如果只有一个src会出问题
class TrajDatasetGeneral(torch.utils.data.Dataset):
    def __init__(self, src):
        super(TrajDatasetGeneral, self).__init__()
        assert type(src) is list
        self.src = src
        return

    def __len__(self):
        return self.src[0].shape[0]

    def __getitem__(self, idx):
        return [_src[idx] for _src in self.src]

    def get_sample(self, idx):
        assert type(idx) is int
        data = self.__getitem__(idx)
        return [torch.tensor(_dt.reshape(1, -1)) for _dt in data]


def wrap_data(batch_data, keys, device=None):
    if device is None:
        batch_data = [_data.T for _data in batch_data]
    else:
        batch_data = [_data.T.to(device) for _data in batch_data]

    return dict(zip(keys, batch_data))

