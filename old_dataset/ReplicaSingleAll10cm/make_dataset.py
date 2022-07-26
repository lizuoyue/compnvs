import os, h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

import torch
import sys, tqdm
sys.path.append("..")

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

def voxelize_points(pts, info, voxel_size, ref=np.array([0, 0, 0])):
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

    voxel_idx = np.unique(np.floor((pts - ref) / voxel_size).astype(np.int), axis=0) * 2 + 1 # always odd

    vertex_idx = np.repeat(voxel_idx, 8, axis=0) + np.tile(offset, (voxel_idx.shape[0], 1)) # all voxel vertices index, always even
    vertex_idx, voxel_to_vertex = np.unique(vertex_idx, axis=0, return_inverse=True)
    voxel_to_vertex = voxel_to_vertex.reshape((-1, 8))

    voxel_pts = voxel_idx / 2.0 * voxel_size + ref # voxel coordinates
    vertex_pts = vertex_idx / 2.0 * voxel_size + ref # vertex coordinates

    num_c = info.shape[1]
    vertex_info = np.zeros((vertex_idx.shape[0], num_c + 1))

    vertex_idx_to_1d_idx = {}
    for d_idx, line in enumerate(vertex_idx):
        x, y, z = line
        vertex_idx_to_1d_idx[(x, y, z)] = d_idx

    pt_to_voxel = np.round((pts - ref) / voxel_size).astype(np.int) * 2 # always even
    for (x, y, z), pt_info in zip(pt_to_voxel, info):
        idx = vertex_idx_to_1d_idx[(x, y, z)]
        vertex_info[idx, :num_c] += pt_info
        vertex_info[idx, num_c] += 1

    return voxel_pts, voxel_to_vertex, (vertex_idx / 2.0).astype(np.int), vertex_info

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

    # os.system('mkdir all')
    # os.system('mkdir all/pose')
    # os.system('mkdir all/pose_identical')
    # os.system('mkdir all/depth')
    # os.system('mkdir all/rgb')
    os.system('mkdir all/npz')

    # with open('all/intrinsics.txt', 'w') as f:
    #     f.write('128.0 0.0 128.0 0.0\n')
    #     f.write('0.0 128.0 128.0 0.0\n')
    #     f.write('0.0 0.0 1.0 0.0\n')
    #     f.write('0.0 0.0 0.0 1.0\n')
    
    voxel_size = 0.1
    half_voxel_size = voxel_size / 2.0
    mask = np.ones((256, 256)).astype(np.uint8)
    mask[:,(128-32):(128+32)] = 0
    # mask     1 | 0 | 1

    for val, phase in enumerate(['train']): # , 'val'

        h5_filename = f'/home/lzq/lzy/mono_dep_novel_view/dataset/{phase}_replica_pose.h5'
        h5_file = h5py.File(h5_filename, 'r')

        for i in tqdm.tqdm(list(range(0, 250))): # 250

            sid = i // 250

            rgb = h5_file['rgb'][(i)]
            dep = h5_file['dep'][(i)]
            pose = pose2rotmat(h5_file['pose'][(i)][np.newaxis])[0]

            gt_rgb = rgb.copy().astype(np.float)
            gt_rgb = gt_rgb / 127.5 - 1.0

            rgbm = gt_rgb.copy() 
            rgbm[mask == 0] = np.array([0, 0, 0])
            rgbm = np.concatenate([rgbm, mask[..., np.newaxis].astype(np.float)], axis=-1)

            local_pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), np.eye(4))
            # local_pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), pose)
            
            gt_rgb = rgb.reshape((-1, 3))
            rgbm = rgbm.reshape((-1, 4))
            local_pts = local_pts.reshape((-1, 3))

            reference = local_pts.min(axis=0) - half_voxel_size

            valid = (dep[..., 0] > 0).reshape((-1))

            rgbm = rgbm[valid]
            gt_rgb = gt_rgb[valid]
            local_pts = local_pts[valid]

            voxel_pts, voxel_to_vertex, vertex_idx, vertex_info = voxelize_points(local_pts, rgbm, voxel_size, reference)
            print(vertex_idx.min(axis=0))
            print(vertex_idx.max(axis=0))

            # f1 = open('temp1.txt', 'w')
            # f2 = open('temp2.txt', 'w')
            # f3 = open('temp3.txt', 'w')
            # for (x, y, z), (r, g, b, m, n) in zip(vertex_idx * voxel_size, vertex_info):
            #     if m > 0: # m > 0, n > 0, meaning the vertex has colored points
            #         r = round((r / m + 1) * 127.5)
            #         g = round((g / m + 1) * 127.5)
            #         b = round((b / m + 1) * 127.5)
            #         f1.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, r, g, b))
            #     elif n > 0: # m = 0, n > 0, meaning the vertex has only uncolored points
            #         f2.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, 127, 127, 127))
            #     else: # m = 0, n = 0, meaning the vertex has no points
            #         f3.write('%.6lf;%.6lf;%.6lf;%d;%d;%d\n' % (x, y, z, 0, 0, 0))
            # f1.close()
            # f2.close()
            # f3.close()
            # quit()

            vertex_info[..., :3] = vertex_info[..., :3] / np.maximum(vertex_info[..., 3:4], 1)
            vertex_info[..., 3:4] = vertex_info[..., 3:4] / np.maximum(vertex_info[..., 4:5], 1)
            vertex_info = vertex_info[..., :4]

            # Image.fromarray(rgb).save(f'all/rgb/{val}_{i:05d}.png')
            # Image.fromarray(dep[..., 0].astype(np.uint8)).save(f'all/depth/{val}_{i:05d}.png')

            # with open(f'all/true_pose/{val}_{i:05d}.txt', 'w') as f:
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[0])))
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[1])))
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[2])))
            #     f.write('0.0 0.0 0.0 1.0\n')
            
            # with open(f'all/pose/{val}_{i:05d}.txt', 'w') as f:
            #     f.write('1.0 0.0 0.0 0.0\n')
            #     f.write('0.0 1.0 0.0 0.0\n')
            #     f.write('0.0 0.0 1.0 0.0\n')
            #     f.write('0.0 0.0 0.0 1.0\n')

            np.savez_compressed(
                f'all/npz/{val}_{i:05d}.npz',
                voxel_pts=voxel_pts.astype(np.float32),
                voxel_to_vertex=voxel_to_vertex,
                vertex_idx=vertex_idx,
                vertex_info=vertex_info,
                reference=reference,
            )


