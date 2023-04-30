import numpy as np
from dataloader.sprite import Sprite

def load_dataset(args):

    path = args.dataset_path
    with open(path + 'sprites_X_train.npy', 'rb') as f:
        X_train = np.load(f)
    with open(path + 'sprites_X_test.npy', 'rb') as f:
        X_test = np.load(f)
    with open(path + 'sprites_A_train.npy', 'rb') as f:
        A_train = np.load(f)
    with open(path + 'sprites_A_test.npy', 'rb') as f:
        A_test = np.load(f)
    with open(path + 'sprites_D_train.npy', 'rb') as f:
        D_train = np.load(f)
    with open(path + 'sprites_D_test.npy', 'rb') as f:
        D_test = np.load(f)

    train_data = Sprite(data=X_train, A_label=A_train, D_label=D_train)
    test_data = Sprite(data=X_test, A_label=A_test, D_label=D_test)

    return train_data, test_data