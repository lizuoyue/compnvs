import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points

import numpy as np
import tqdm, random
from PIL import Image
import os, glob
from os.path import join as opj
from utils import project_depth

if __name__ == '__main__':

    data_src_dir = '/home/lzq/lzy/replica_habitat_gen/'
    data_dst_dir = 'gen_data_pytorch3d_mid'
    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])
    bboxes = np.loadtxt(opj(data_src_dir, 'ReplicaGen/bboxes.txt'))
    img_size = 256

    for sid in [13,14,19,20,21,42]:#range(1,48):
        bbox = bboxes[sid]
        triplets = np.load(opj(data_src_dir, f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz'))['mid']

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

            uv, d = project_depth(in_pc, int_mat, k_pose)
            uv = (uv / -img_size) * 2.0 + 1.0

            pts = torch.from_numpy(np.dstack([uv[:,0], uv[:,1], d])).float().reshape(-1, 3).cuda()
            fts = torch.from_numpy(in_rgb[...,:3]).float().reshape(-1, 3).cuda()

            radius = 4.0 / float(img_size) * 2.0

            pts3D = Pointclouds(points=[pts], features=[fts])
            points_idx, _, dist = rasterize_points(
                pts3D, (img_size, img_size), radius, 8
            )
            

            alphas = (1 - dist.clamp(max=1, min=1e-3).pow(0.5)).permute(0, 3, 1, 2)

            results = compositing.alpha_composite(
                points_idx.permute(0, 3, 1, 2).long(),
                alphas,
                pts3D.features_packed().permute(1,0),
            )

            im = np.round(results[0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            Image.fromarray(im).save(opj(scene_dir, f'{k:03d}_{i:03d}_{j:03d}_in.jpg'))

            mask = ((points_idx[0] >= 0).any(dim=-1).int() * 255).cpu().numpy().astype(np.uint8)
            Image.fromarray(mask).save(opj(scene_dir, f'{k:03d}_{i:03d}_{j:03d}_mask.jpg'))
