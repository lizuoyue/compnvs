import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

if __name__ == '__main__':

    # var, niter = 'simple', 3150
    # var, niter = 'multi1', 8653
    # var, niter = 'df', 6300
    var, niter = 'dfv2', 9448

    with open(f'torch/log_{var}.out') as f:
        lines = f.readlines()
        train_lines = [line.replace(',', '') for line in lines if line.startswith('Epoch')]
        val_lines = [line.replace(',', '') for line in lines if line.startswith('Val ')]
    
    train = []
    for line in train_lines:
        items = line.split()
        if var.startswith('df'):
            epoch, it, _, loss_df, loss, recall = int(items[1]), int(items[3]), float(items[4]), float(items[5]), float(items[6]), float(items[7])
        else:
            epoch, it, loss, recall = int(items[1]), int(items[3]), float(items[4]), float(items[5])
        train.append(np.array([epoch * niter + it, loss, recall]))
    train = np.stack(train)

    val = []
    for i in range(len(val_lines) // 5):
        _, _, epoch, _, it = val_lines[i * 5].split()
        epoch, it = int(epoch), int(it)
        loss = float(val_lines[i * 5 + 1].split()[4]) # 4 geo 3 df
        recall = float(val_lines[i * 5 + 4].split()[2])
        val.append(np.array([epoch * niter + it, loss, recall]))
    val = np.stack(val)

    plt.plot(train[:,0], train[:,1])
    plt.plot(val[:,0], val[:,1])
    plt.plot(train[:,0], train[:,2])
    plt.plot(val[:,0], val[:,2])
    plt.savefig(f'loss_{var}.png')
    plt.clf()

    train_k = np.ones((200, 1))
    train_k /= train_k.sum()
    val_k = np.ones((10, 1))
    val_k /= val_k.sum()

    train = scipy.signal.convolve2d(train, train_k, mode='valid')
    val = scipy.signal.convolve2d(val, val_k, mode='valid')

    plt.plot(train[:,0], train[:,1])
    plt.plot(val[:,0], val[:,1])
    plt.plot(train[:,0], train[:,2])
    plt.plot(val[:,0], val[:,2])
    plt.savefig(f'loss_{var}_mavg.png')
