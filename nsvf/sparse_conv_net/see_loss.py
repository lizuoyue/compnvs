import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

if __name__ == '__main__':

    prefix = 'lzq_unet_gen_ftio'
    with open(f'{prefix}.out') as f:
        lines = f.readlines()
        train_lines = [line for line in lines if line.startswith('Epoch') and 'disc' not in line]
        val_lines = [line for line in lines if line.startswith('Val ')]

    # prefix = 'lzq_unet_gen_ftio'

    # with open(f'{prefix}.out') as f:
    #     lines = f.readlines()
    #     train_lines += [line for line in lines if line.startswith('Epoch')]
    #     val_lines += [line for line in lines if line.startswith('Val ')]
    
    train = []
    for line in train_lines:
        items = line.split()
        epoch, it, loss = int(items[1]), int(items[3]), float(items[4])
        train.append(np.array([epoch * 9438 + it, loss]))
    train = np.stack(train)

    val = []
    for i in range(len(val_lines) // 2):
        items = val_lines[i * 2].split()
        epoch, it = int(items[2]), int(items[4])
        loss = float(val_lines[i * 2 + 1].split()[2])
        val.append(np.array([epoch * 9438 + it, loss]))
    val = np.stack(val)

    plt.plot(train[:,0], train[:,1])
    plt.plot(val[:,0], val[:,1])
    plt.ylim(0, 0.5)
    plt.xlim(0, 60000)
    plt.savefig(f'{prefix}.png')
    plt.clf()

    train_k = np.ones((100, 1))
    train_k /= train_k.sum()
    val_k = np.ones((5, 1))
    val_k /= val_k.sum()

    train = scipy.signal.convolve2d(train, train_k, mode='valid')
    val = scipy.signal.convolve2d(val, val_k, mode='valid')

    plt.plot(train[:,0], train[:,1])
    plt.plot(val[:,0], val[:,1])
    plt.ylim(0, 0.5)
    plt.xlim(0, 60000)
    plt.savefig(f'{prefix}_mavg.png')
