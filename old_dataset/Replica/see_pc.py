import os, h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def unproject(dep, int_mat, ext_mat):
    assert(len(dep.shape) == 2)
    h, w = dep.shape
    x, y = np.meshgrid(np.arange(w).astype(np.float32)+0.5, np.arange(h).astype(np.float32)+0.5)
    z = np.ones((h, w), np.float32)
    pts = np.stack([x, y, z], axis=-1)
    pts = pts.dot(np.linalg.inv(int_mat).T) # local
    pts = pts * np.stack([dep] * 3, axis=-1)
    pts = np.concatenate([pts, np.ones((h, w, 1), np.float32)], axis=-1)
    pts = pts.dot(ext_mat.T)[..., :3]
    return pts


if __name__ == '__main__':

    # Replica Office 3

    bbox = np.loadtxt('office3/bbox.txt')
    intrinsics = np.loadtxt('office3/intrinsics.txt')[:3,:3]

    all_pts = []
    for i in [0,3,6,9,12]:

        color = np.array(Image.open(f'office3/rgb/0_{i:04d}.png'))
        depth = np.array(Image.open(f'office3/depth/0_{i:04d}.png')) / 255.0 * 10.0
        pose = np.loadtxt(f'office3/pose/0_{i:04d}.txt')

        pts = unproject(depth, intrinsics, pose)
        
        all_pts.append(np.concatenate([
            pts.reshape((-1, 3))[::4].astype(np.float32),
            color.reshape((-1, 3))[::4].astype(np.float32)
        ], axis=-1))

    all_pts = np.concatenate(all_pts, axis=0)
    with open('replica_office3_pc.txt', 'w') as f:
        for pt in all_pts:
            f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % tuple(list(pt)))
