import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.signal

if __name__ == '__main__':

    prefix = 'lzq_unet_gen_ftio_regdim4emb32in'
    with open(f'{prefix}.out') as f:
        lines = f.readlines()
        train_lines = [line for line in lines if line.startswith('Train') and 'Gen:' in line]
        val_lines = [line for line in lines if line.startswith('Val') and 'Gen:' in line]

    train = []
    for line in train_lines:
        line_parts = line.strip().split('Gen: ')
        epoch_infos = line_parts[0].split()
        epoch, it = int(epoch_infos[2]), int(epoch_infos[4])

        loss_items = (line_parts[1].split(', ') + [0, ])[:4]
        sum_loss, gen_loss, gen_loss_mask, disc_loss = [float(item) for item in loss_items]
        train.append(np.array([epoch * 8315 + it, gen_loss, gen_loss_mask, disc_loss]))
    train = np.stack(train)

    val = []
    it = 0
    for line in val_lines:
        loss_items = (line.strip().split('Gen: ')[-1].split(', ') + [0, ])[:4]
        sum_loss, gen_loss, gen_loss_mask, disc_loss = [float(item) for item in loss_items]
        val.append(np.array([it, gen_loss, gen_loss_mask, disc_loss]))
        it += 20
    val = np.stack(val)

    plt.plot(train[:,0], train[:,1], label='train_gen_loss')
    plt.plot(train[:,0], train[:,2], label='train_gen_mask_loss')
    plt.plot(train[:,0], train[:,3], label='train_disc_loss')
    plt.plot(val[:,0], val[:,1], label='val_gen_loss')
    plt.plot(val[:,0], val[:,2], label='val_gen_mask_loss')
    plt.plot(val[:,0], val[:,3], label='val_disc_loss')
    plt.ylim(0, 1.5)
    # plt.xlim(0, 60000)
    plt.legend()
    plt.savefig(f'{prefix}.png')
    plt.clf()

    train_k = np.ones((100, 1))
    train_k /= train_k.sum()
    val_k = np.ones((5, 1))
    val_k /= val_k.sum()

    train = scipy.signal.convolve2d(train, train_k, mode='valid')
    val = scipy.signal.convolve2d(val, val_k, mode='valid')

    plt.plot(train[:,0], train[:,1], label='train_gen_loss')
    plt.plot(train[:,0], train[:,2], label='train_gen_mask_loss')
    plt.plot(train[:,0], train[:,3], label='train_disc_loss')
    plt.plot(val[:,0], val[:,1], label='val_gen_loss')
    plt.plot(val[:,0], val[:,2], label='val_gen_mask_loss')
    plt.plot(val[:,0], val[:,3], label='val_disc_loss')
    plt.ylim(0, 1.5)
    # plt.xlim(0, 60000)
    plt.legend()
    plt.savefig(f'{prefix}_mavg.png')
