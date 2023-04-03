import numpy as np
import os, json, tqdm
from PIL import Image

def project(pts, int_mat, ext_mat, valid=(0, 256, 0, 256)):
    assert(len(pts.shape) == 2) # (N, 3)
    local = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=-1)
    local = local.dot(np.linalg.inv(ext_mat).T)[:, :3]
    local = local.dot(int_mat.T)
    local[:, :2] /= local[:, 2:]
    a, b, c, d = valid
    valid = np.ones((pts.shape[0],), np.bool)
    valid = valid & (local[:, 0] >= a)
    valid = valid & (local[:, 0] <= b)
    valid = valid & (local[:, 1] >= c)
    valid = valid & (local[:, 1] <= d)
    valid = valid & (local[:, 2] > 0)
    return valid

if __name__ == '__main__':

    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])

    phase, n_seq = 'train', 101
    # phase, n_seq = 'test', 10
    relations = []

    for sid in tqdm.tqdm(list(range(n_seq))):
        center_coord = np.loadtxt(f'ReplicaGSN/scene{sid:03d}/init_voxel/center_points_0.1.txt')
        valid_li = []
        for fid in tqdm.tqdm(list(range(100))):
            ext_mat = np.loadtxt(f'ReplicaGSN/scene{sid:03d}/pose/{fid:02d}.txt')
            valid_li.append(project(center_coord, int_mat, ext_mat))
        
        res = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                res[i, j] = (valid_li[i] & valid_li[j]).sum() / valid_li[j].sum()
        
        relations.append(res)
    
    relations = np.stack(relations)
    np.savez_compressed(f'{phase}_relations.npz', relations=relations)
