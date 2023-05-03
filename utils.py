import torch
import numpy as np
from dataloader.sprite import Sprite
import matplotlib.pyplot as plt


def t_to_np(X):
    if X.dtype in [torch.float32, torch.float64]:
        X = X.detach().cpu().numpy()
    return X

def np_to_t(X, device='cuda'):
    if torch.cuda.is_available() is False:
        device = 'cpu'

    from numpy import dtype
    if X.dtype in [dtype('float32'), dtype('float64')]:
        X = torch.from_numpy(X.astype(np.float32)).to(device)
    return X

def load_checkpoint(model, checkpoint_name):
    print("Loading Checkpoint from '{}'".format(checkpoint_name))
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['state_dict'])

def imshow_seqeunce(DATA, plot=True, titles=None, figsize=(50, 10), fontsize=50):
    rc = 2 * len(DATA[0])
    fig, axs = plt.subplots(rc, 2, figsize=figsize)

    for ii, data in enumerate(DATA):
        for jj, img in enumerate(data):

            img = t_to_np(img)
            tsz, csz, hsz, wsz = img.shape
            img = img.transpose((2, 0, 3, 1)).reshape((hsz, tsz * wsz, -1))

            ri, ci = jj * 2 + ii // 2, ii % 2
            axs[ri][ci].imshow(img)
            axs[ri][ci].set_axis_off()
            if titles is not None:
                axs[ri][ci].set_title(titles[ii][jj], fontsize=fontsize)

    plt.subplots_adjust(wspace=.05, hspace=0)

    if plot:
        plt.show()

def reorder(sequence):
    return sequence.permute(0, 1, 4, 2, 3)

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