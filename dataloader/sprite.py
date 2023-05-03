import numpy as np
from torch.utils.data import Dataset


class Sprite(Dataset):
    def __init__(self, data, A_label, D_label):
        self.data = data
        self.A_label = A_label
        self.D_label = D_label
        self.N = self.data.shape[0]

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        data = self.data[index] # (8, 64, 64, 3)
        A_label = self.A_label[index] # (4,)
        D_label = self.D_label[index] # ()

        return {"images": data, "A_label": A_label, "D_label": D_label, "index": index}
