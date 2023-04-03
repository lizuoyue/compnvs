import tqdm, torch
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

class PseudoColorConverter:
    def __init__(self, cmap_name, start_val=-0.15, stop_val=0.15):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        return

    def convert(self, val):
        return np.round(self.scalarMap.to_rgba(val)[..., :3] * 255).astype(np.uint8)

def get_std(filename, num):
    ckpt = torch.load(filename)
    fts = []
    for i in range(num):
        fts.append(ckpt['model'][f'encoder.all_voxels.{i}.values.weight'].numpy().astype(np.float32))
    fts = np.concatenate(fts, axis=0)
    return fts.std(axis=0).mean()

if __name__ == '__main__':

    num, num_train, num_val = [], [], []
    for i in range(1, 7):
        num.append(get_std(f'/home/lzq/lzy/NSVF/ReplicaGen/multi_scene_nsvf_basev1/checkpoint{i*10}.pt', 42))
    print(num)

    for i in range(1, 7):
        i = min(i*7, 40)
        num_train.append(get_std(f'/home/lzq/lzy/NSVF/ReplicaGen/multi_scene_nsvf_basev1_train/checkpoint{i}.pt', 42))
    print(num_train)

    for i in range(1, 7):
        num_val.append(get_std(f'/home/lzq/lzy/NSVF/ReplicaGen/multi_scene_nsvf_basev1_val/checkpoint{i}.pt', 6))
    print(num_val)

    plt.plot(np.arange(1,7), num, '-')
    plt.plot(np.arange(1,7), num_val, '-')
    plt.plot(np.arange(1,7), num_train, '-')
    plt.savefig(f'std.png')
    plt.clf()
