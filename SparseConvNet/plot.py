import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

def get_loss(filename):
    loss = np.loadtxt(filename)
    loss = loss[:(loss.shape[0] // 45 * 45), 1].reshape((-1, 45)).mean(axis=1)
    return loss

if __name__ == '__main__':

    unet_loss = get_loss('unet.out')
    gated_unet_loss = get_loss('gated_unet.out')
    unet_plus_loss = get_loss('unet_plus.out')

    print(unet_loss[4500])
    print(unet_plus_loss[4500])

    plt.plot(unet_loss)
    plt.plot(unet_plus_loss)
    # plt.plot(gated_unet_loss)
    plt.savefig('plot.png')

