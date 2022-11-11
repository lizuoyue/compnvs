import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

def lines2arr(lines):
    li = []
    for i, line in enumerate(lines):
        parts = line.strip().split()
        epoch, it = int(parts[2]), int(parts[4][:-1])
        li.append(np.array([i] + [float(item) for item in parts[5:]]))
    return np.stack(li)

if __name__ == '__main__':

    prefix = 'unet_extrap'#'pconv_unet50_encft'#'unet34c_rgba_init'
    with open(f'{prefix}.out') as f:
        lines = f.readlines()
        train_lines = [line for line in lines if line.startswith('Train Epoch')]
        val_lines = [line for line in lines if line.startswith('Val Epoch')]

    train = lines2arr(train_lines)
    val = lines2arr(val_lines)
    val[:,0] *= 20

    plt.plot(train[:,0], train[:,1], label='train_dis_loss')
    plt.plot(train[:,0], train[:,2], label='train_gen_gan_loss')
    plt.plot(train[:,0], train[:,3], label='train_gen1_loss')
    plt.plot(train[:,0], train[:,4], label='train_gen2_loss')
    plt.plot(val[:,0], val[:,1], label='val_dis_loss')
    plt.plot(val[:,0], val[:,2], label='val_gen_gan_loss')
    plt.plot(val[:,0], val[:,3], label='val_gen1_loss')
    plt.plot(val[:,0], val[:,4], label='val_gen2_loss')
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

    plt.plot(train[:,0], train[:,1], label='train_dis_loss')
    plt.plot(train[:,0], train[:,2], label='train_gen_gan_loss')
    plt.plot(train[:,0], train[:,3], label='train_gen1_loss')
    plt.plot(train[:,0], train[:,4], label='train_gen2_loss')
    plt.plot(val[:,0], val[:,1], label='val_dis_loss')
    plt.plot(val[:,0], val[:,2], label='val_gen_gan_loss')
    plt.plot(val[:,0], val[:,3], label='val_gen1_loss')
    plt.plot(val[:,0], val[:,4], label='val_gen2_loss')
    # plt.ylim(0, 1.0)
    # plt.xlim(0, 60000)
    plt.legend()
    plt.title(prefix)
    plt.savefig(f'{prefix}_mean_avg.png')
