import os, h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

import torch
import sys, tqdm
sys.path.append("..")
from fairnr.data.geometry import offset_points, discretize_points

def pose2rotmat(pose):
    # Pose: B x 7
    b, n = pose.shape
    assert(n == 7)
    r = R.from_quat(pose[:,[4,5,6,3]])
    rotmat = r.as_matrix() # B x 3 x 3
    rotmat = np.concatenate([rotmat, pose[:,:3,np.newaxis]], axis=2) # B x 3 x 4
    to_cat = np.zeros((b, 1, 4))
    to_cat[:,:,-1] = 1
    rotmat = np.concatenate([rotmat, to_cat], axis=1) # B x 4 x 4
    # Replica coordinates
    neg_yz = np.diag([1.0,-1.0,-1.0,1.0]).astype(np.float32)
    return rotmat.astype(np.float32).dot(neg_yz)

def voxelize_points(pts, info, voxel_size):
    # pts: shape of N, 3
    # info: shape of N, C
    # return voxel centers
    half_voxel = voxel_size / 2.0
    offset = np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int)

    voxel_idx = np.unique(np.floor(pts / voxel_size).astype(np.int), axis=0) * 2 # always even, ~ 20000 points

    vertex_idx = np.repeat(voxel_idx, 8, axis=0) + np.tile(offset, (voxel_idx.shape[0], 1)) # all voxel vertices index
    vertex_idx, voxel_to_vertex = np.unique(vertex_idx, axis=0, return_inverse=True)
    voxel_to_vertex = voxel_to_vertex.reshape((-1, 8))

    voxel_pts = (voxel_idx + 1) / 2.0 * voxel_size # voxel coordinates
    vertex_pts = (vertex_idx + 1) / 2.0 * voxel_size # vertex coordinates

    num_c = info.shape[1]
    vertex_info = np.zeros((vertex_idx.shape[0], num_c + 1))

    odd_idx_to_1d_idx = {}
    for d_idx, line in enumerate(vertex_idx):
        x, y, z = line
        odd_idx_to_1d_idx[(x, y, z)] = d_idx

    pt_to_voxel = np.floor((pts - half_voxel) / voxel_size).astype(np.int) * 2 + 1 # always odd
    for (x, y, z), pt_info in zip(pt_to_voxel, info):
        idx = odd_idx_to_1d_idx[(x, y, z)]
        vertex_info[idx, :num_c] += pt_info
        vertex_info[idx, num_c] += 1

    return voxel_pts, voxel_to_vertex, ((vertex_idx + 1) / 2.0).astype(np.int), vertex_info

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

    # Office 3
    # quit()

    os.system('mkdir all')
    os.system('mkdir all/pose')
    os.system('mkdir all/true_pose')
    os.system('mkdir all/depth')
    os.system('mkdir all/rgb')
    os.system('mkdir all/npz')

    with open('all/intrinsics.txt', 'w') as f:
        f.write('128.0 0.0 128.0 0.0\n')
        f.write('0.0 128.0 128.0 0.0\n')
        f.write('0.0 0.0 1.0 0.0\n')
        f.write('0.0 0.0 0.0 1.0\n')
    
    voxel_size = 0.05
    half_voxel_size = voxel_size / 2.0
    mask = np.ones((256, 256)).astype(np.uint8)
    mask[:,(128-32):(128+32)] = 0

    i = 0

    for val, phase in enumerate(['train', 'val']):

        h5_filename = f'/home/lzq/lzy/mono_dep_novel_view/dataset/{phase}_replica_pose.h5'
        h5_file = h5py.File(h5_filename, 'r')
        for rgb, dep, pose in tqdm.tqdm(list(zip(h5_file['rgb'][()], h5_file['dep'][()], pose2rotmat(h5_file['pose'][()])))):

            gt_rgb = rgb.copy().astype(np.float)
            gt_rgb = gt_rgb / 127.5 - 1.0

            rgbm = gt_rgb.copy() 
            rgbm[mask == 0] = np.array([0, 0, 0])
            rgbm = np.concatenate([rgbm, mask[..., np.newaxis].astype(np.float)], axis=-1)

            local_pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), np.eye(4))
            
            gt_rgb = rgb.reshape((-1, 3))
            rgbm = rgbm.reshape((-1, 4))
            local_pts = local_pts.reshape((-1, 3))

            valid = (dep[..., 0] > 0).reshape((-1))

            rgbm = rgbm[valid]
            gt_rgb = gt_rgb[valid]
            local_pts = local_pts[valid]

            voxel_pts, voxel_to_vertex, vertex_idx, vertex_info = voxelize_points(local_pts, rgbm, voxel_size)

            # with open('temp.txt', 'w') as f:
            #     for (x,y,z),(r,g,b,m,n) in zip(vertex_idx * voxel_size, vertex_info):
            #         if n > 0:
            #             r=round((r/n+1)*127.5)
            #             g=round((g/n+1)*127.5)
            #             b=round((b/n+1)*127.5)
            #             f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x,y,z,r,g,b))
            # quit()

            vertex_info = vertex_info[..., :-1] / np.maximum(vertex_info[..., -1:], 1)

            Image.fromarray(rgb).save(f'all/rgb/{val}_{i:05d}.png')
            Image.fromarray(dep[..., 0].astype(np.uint8)).save(f'all/depth/{val}_{i:05d}.png')

            with open(f'all/true_pose/{val}_{i:05d}.txt', 'w') as f:
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[0])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[1])))
                f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[2])))
                f.write('0.0 0.0 0.0 1.0\n')
            
            with open(f'all/pose/{val}_{i:05d}.txt', 'w') as f:
                f.write('1.0 0.0 0.0 0.0\n')
                f.write('0.0 1.0 0.0 0.0\n')
                f.write('0.0 0.0 1.0 0.0\n')
                f.write('0.0 0.0 0.0 1.0\n')

            np.savez_compressed(
                f'all/npz/{val}_{i:05d}.npz',
                voxel_pts=voxel_pts.astype(np.float32),
                voxel_to_vertex=voxel_to_vertex,
                vertex_idx=vertex_idx,
                vertex_info=vertex_info,
            )
            i += 1
        


