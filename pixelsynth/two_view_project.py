import numpy as np
import tqdm, random
from PIL import Image
import os, glob
from os.path import join as opj
from utils import unproject, filter_points, project, project_depth

if __name__ == '__main__':

    data_src_dir = '/home/lzq/lzy/replica_habitat_gen/'
    data_dst_dir = 'gen_data'
    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])
    bboxes = np.loadtxt(opj(data_src_dir, 'ReplicaGen/bboxes.txt'))

    for sid in range(48):
        bbox = bboxes[sid]
        triplets = np.load(opj(data_src_dir, f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz'))['easy']

        scene_dir = opj(data_dst_dir, f'{sid:02d}')
        os.makedirs(scene_dir, exist_ok=True)

        for k, i, j in tqdm.tqdm(triplets):
            i_data = np.load(opj(data_src_dir, f'ReplicaGenFt/all/npz/{sid:02d}_{i:03d}.npz'))
            j_data = np.load(opj(data_src_dir, f'ReplicaGenFt/all/npz/{sid:02d}_{j:03d}.npz'))
            k_data = np.load(opj(data_src_dir, f'ReplicaGenFt/all/npz/{sid:02d}_{k:03d}.npz'))

            i_pose = np.loadtxt(opj(data_src_dir, f'ReplicaGenFt/all/pose/{sid:02d}_{i:03d}.txt'))
            j_pose = np.loadtxt(opj(data_src_dir, f'ReplicaGenFt/all/pose/{sid:02d}_{j:03d}.txt'))
            k_pose = np.loadtxt(opj(data_src_dir, f'ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt'))

            in_pc = np.concatenate([i_data['pts'], j_data['pts']], axis=0)
            in_rgb = np.concatenate([i_data['rgb'], j_data['rgb']], axis=0)

            k_in = np.zeros((256, 256, 3), np.uint8)

            valid_mask = project(in_pc, int_mat, k_pose)
            uv, d = project_depth(in_pc, int_mat, k_pose)
            uv = np.floor(uv).astype(np.int32)
            d = d[valid_mask]
            order = np.argsort(d)[::-1]
            d = d[order]
            uv = uv[valid_mask][order]
            rgb = in_rgb[valid_mask][order]

            k_in[uv[:,1], uv[:,0]] = rgb
            k_in_img = Image.fromarray(np.uint8(k_in))
            k_in_img.save(opj(scene_dir, f'{k:03d}_{i:03d}_{j:03d}_in.jpg'))

            k_mask = np.zeros((256, 256), np.uint8)
            k_mask[uv[:,1], uv[:,0]] = 255
            k_mask_img = Image.fromarray(np.uint8(k_mask))
            k_mask_img.save(opj(scene_dir, f'{k:03d}_{i:03d}_{j:03d}_mask.jpg'))