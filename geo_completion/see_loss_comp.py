import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

def lines2arr(lines):
    li = []
    for i, line in enumerate(lines):
        parts = line.replace(', ', ',').strip().split()
        epoch, it = int(parts[2]), int(parts[4])
        li.append(np.array([i] + eval(parts[5]) + list(eval(parts[6]))))
    return np.stack(li)

if __name__ == '__main__':

    prefix = 'comp_crop'
    with open(f'{prefix}.out') as f:
        lines = f.readlines()[:10000]
        train_lines = [line for line in lines if line.startswith('Train Epoch')]
        val_lines = [line for line in lines if line.startswith('Val Epoch')]

    train = lines2arr(train_lines)
    val = lines2arr(val_lines)
    val[:,0] *= 20

    for data, name in zip([train, val], ['train', 'val']):
        for i in range(6,7):
            plt.plot(data[:,0], data[:,i], label=f'{name}_bce_loss_{i}')
        plt.plot(data[:,0], data[:,7], label=f'{name}_precision')
        plt.plot(data[:,0], data[:,8], label=f'{name}_recall')
        plt.plot(data[:,0], data[:,9], label=f'{name}_iou')
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 60000)
    plt.legend()
    plt.title(prefix)
    plt.savefig(f'{prefix}.png')
    plt.clf()

    train_k = np.ones((100, 1))
    train_k /= train_k.sum()
    val_k = np.ones((5, 1))
    val_k /= val_k.sum()

    train = scipy.signal.convolve2d(train, train_k, mode='valid')
    val = scipy.signal.convolve2d(val, val_k, mode='valid')

    for data, name in zip([train, val], ['train', 'val']):
        for i in range(6,7):
            plt.plot(data[:,0], data[:,i], label=f'{name}_bce_loss_{i}')
        plt.plot(data[:,0], data[:,7], label=f'{name}_precision')
        plt.plot(data[:,0], data[:,8], label=f'{name}_recall')
        plt.plot(data[:,0], data[:,9], label=f'{name}_iou')
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 60000)
    plt.legend()
    plt.title(prefix)
    plt.savefig(f'{prefix}_mean_avg.png')
