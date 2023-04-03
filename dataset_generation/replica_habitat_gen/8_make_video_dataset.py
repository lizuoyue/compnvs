import numpy as np
import tqdm, random
from PIL import Image
import torch
import os, glob
from utils import interplate_poses

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

if __name__ == '__main__':

    # encoder.all_voxels.{sid}.points
    # encoder.all_voxels.{sid}.keys XXXXXX
    # encoder.all_voxels.{sid}.feats
    # encoder.all_voxels.{sid}.num_keys XXXXXX
    # encoder.all_voxels.{sid}.keep
    # encoder.all_voxels.{sid}.voxel_size XXXXXX
    # encoder.all_voxels.{sid}.step_size XXXXXX
    # encoder.all_voxels.{sid}.max_hits XXXXXX
    # encoder.all_voxels.{sid}.values.weight

    n_scene = 13
    voxel_size = 0.1
    spatial_size = [144, 64, 160] # center

    for sid in [14,19,20,21,42]:#range(n_scene, n_scene+1):

        files = sorted(glob.glob(f'ReplicaGenEncFtTriplets/mid/npz_pred/{sid:02d}_*.npz'))
        assert(len(files) > 0)

        for file in tqdm.tqdm(files):
            basename = os.path.basename(file).replace('.npz', '')
            _, k, i, j = [int(item) for item in basename.split('_')]

            pose_i = np.loadtxt(f'ReplicaGenFt/all/pose/{sid:02d}_{i:03d}.txt')
            pose_j = np.loadtxt(f'ReplicaGenFt/all/pose/{sid:02d}_{j:03d}.txt')
            pose_k = np.loadtxt(f'ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt')

            poses_ik = interplate_poses(pose_i, pose_k, num=8)
            poses_kj = interplate_poses(pose_k, pose_j, num=8)
            poses_ikj = np.concatenate([poses_ik, poses_kj[1:]], axis=0)
            
            np.savez_compressed(f'ReplicaGenEncFtTriplets/mid/pose_video/{basename}.npz', poses=poses_ikj)

            # for n in range(15):
            #     with open(f'ReplicaGenTempVideo/all/pose/{basename}_{n:02d}.txt', 'w') as f:
            #         f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(poses_ikj[n, 0])))
            #         f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(poses_ikj[n, 1])))
            #         f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(poses_ikj[n, 2])))
            #         f.write('0.0 0.0 0.0 1.0\n')
            #     if n in [0, 7, 14]:
            #         xxx = (n == 0) * i + (n == 7) * k + (n == 14) * j
            #         src_rgb = f'ReplicaGen/scene{sid:02d}/rgb/{xxx:03d}.png'
            #         tar_rgb = f'ReplicaGenTempVideo/all/rgb/{basename}_{n:02d}.png'
            #         os.system(f'cp {src_rgb} {tar_rgb}')
            #     else:
            #         Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(f'ReplicaGenTempVideo/all/rgb/{basename}_{n:02d}.png')

            # src_npz = file
            # tar_npz = f'ReplicaGenTempVideo/all/npz/{basename}.npz'
            # os.system(f'cp {src_npz} {tar_npz}')
