import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

if __name__ == '__main__':

    prefix = 'log_tdf_focal_lr1e-5_ocinp_new'
    with open(f'{prefix}.out') as f:
        lines = f.readlines()
        train_lines = [line for line in lines if line.startswith('Epoch')]
        val_lines = [line for line in lines if line.startswith('Val ')]

    train = []
    for line in train_lines:
        items = line.strip().split()
        # Epoch 0 Iter 1 17.229843 10.829764 6.400079 0.015 0.015 0.991 [11171 31466 37599 11578]
        epoch, it = int(items[1]), int(items[3])
        df_loss, oc_loss = float(items[5]), float(items[6])
        iou, precision, recall = float(items[7]), float(items[8]), float(items[9])
        train.append(np.array([epoch * 4719 + it, df_loss, oc_loss, iou, precision, recall]))
    train = np.stack(train)

    val = []
    for line in val_lines:
        items = line.strip().split()
        # Val Epoch 1 Iter 7760 2.370984 0.057897 2.313086 0.247610 0.255242 0.892248
        epoch, it = int(items[2]), int(items[4])
        df_loss, oc_loss = float(items[6]), float(items[7])
        iou, precision, recall = float(items[8]), float(items[9]), float(items[10])
        val.append(np.array([epoch * 4719 + it, df_loss, oc_loss, iou, precision, recall]))
    val = np.stack(val)

    # plt.plot(train[:,0], train[:,1])
    # plt.plot(val[:,0], val[:,1])
    # plt.ylim(0, 0.5)
    # plt.xlim(0, 30000)
    # plt.savefig(f'{prefix}.png')
    # plt.clf()

    train_k = np.ones((100, 1))
    train_k /= train_k.sum()
    val_k = np.ones((5, 1))
    val_k /= val_k.sum()

    train = scipy.signal.convolve2d(train, train_k, mode='valid')
    val = scipy.signal.convolve2d(val, val_k, mode='valid')

    plt.plot(train[:,0], train[:,1], label='train:dfl')
    plt.plot(train[:,0], train[:,2], label='train:ocl')
    plt.plot(val[:,0], val[:,1], label='test:dfl')
    plt.plot(val[:,0], val[:,2], label='test:ocl')
    plt.ylim(0, 10)
    plt.legend()
    plt.savefig(f'{prefix}_loss.png')

    plt.clf()
    plt.plot(train[:,0], train[:,3], label='train:IoU')
    plt.plot(train[:,0], train[:,4], label='train:Prec')
    plt.plot(train[:,0], train[:,5], label='train:Recl')
    plt.plot(val[:,0], val[:,3], label='test:IoU')
    plt.plot(val[:,0], val[:,4], label='test:Prec')
    plt.plot(val[:,0], val[:,5], label='test:Recl')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f'{prefix}_metric.png')
