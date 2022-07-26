import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':

    for s in ['', '_l2', '_posenc', '_posenc_l2']:
        with open(f'log{s}.out') as f:
            lines = [[float(item) for item in line.split()[1:]] for line in f.readlines()]
            lines = np.array(lines).T[:,:5000]
            plt.plot(lines[0], lines[1], label=f'observed{s}')
            plt.plot(lines[0], lines[2], '--', label=f'unobserved{s}')
    plt.legend(loc='upper right')
    plt.savefig(f'loss.png')
