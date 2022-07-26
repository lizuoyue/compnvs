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

    os.system('mkdir all')
    os.system('mkdir all/pose')
    os.system('mkdir all/depth')
    os.system('mkdir all/rgb')
    os.system('mkdir all/npz')
    os.system('mkdir all/target_ft')

    with open('all/intrinsics.txt', 'w') as f:
        f.write('128.0 0.0 128.0 0.0\n')
        f.write('0.0 128.0 128.0 0.0\n')
        f.write('0.0 0.0 1.0 0.0\n')
        f.write('0.0 0.0 0.0 1.0\n')
    
    voxel_size = 0.1
    half_voxel_size = voxel_size / 2.0
    mask = np.ones((256, 256)).astype(np.uint8)
    mask[:,(128-32):(128+32)] = 0
    # mask     1 | 0 | 1

    ckpt = torch.load(f'../ReplicaMultiScene/multi_scene_nsvf_basev1_10cm_refine/save/checkpoint_16_78000.pt')
    # ['model']['points', 'feats', 'values.weight']

    for val, phase in enumerate(['train']): # , 'val'

        h5_filename = f'/home/lzq/lzy/mono_dep_novel_view/dataset/{phase}_replica_pose.h5'
        h5_file = h5py.File(h5_filename, 'r')

        for sid in range(45):

            print(sid)
            center_point = ckpt['model'][f'encoder.all_voxels.{sid}.points'].numpy()
            print(center_point.shape)
            center_keep = ckpt['model'][f'encoder.all_voxels.{sid}.keep'].numpy().astype(np.bool)
            print(center_keep.shape, np.unique(center_keep))
            center_to_vertex = ckpt['model'][f'encoder.all_voxels.{sid}.feats'].numpy()
            print(center_to_vertex.shape)
            vertex_feature = ckpt['model'][f'encoder.all_voxels.{sid}.values.weight'].numpy()
            print(vertex_feature.shape)
            vertex_color = np.loadtxt(f'../ReplicaMultiScene/scene{sid:02d}/init_voxel/vertex_0.1_with_rgb.txt', delimiter=';')
            print(vertex_color.shape)
            vertex_info_raw = np.concatenate([
                vertex_color[:, 3:6].astype(np.int32), # color
                (vertex_color[:, 6:] > 0).astype(np.int32), # has RGB or not
                (vertex_color[:, 6:] * 0 + 1).astype(np.int32), # has observation or not
            ], axis=-1)
            print(vertex_info_raw.shape)

            reference = vertex_color[:, :3].min(axis=0).astype(np.float32)
            vertex_3d_idx_raw = (vertex_color[:, :3] - reference) / voxel_size
            vertex_3d_idx = np.round(vertex_3d_idx_raw).astype(np.int32)
            assert(np.abs(vertex_3d_idx - vertex_3d_idx_raw).max() < 1e-3)

            for i in tqdm.tqdm(list(range(sid * 250, sid * 250 + 250))):

                rgb = h5_file['rgb'][(i)]
                dep = h5_file['dep'][(i)]
                pose = pose2rotmat(h5_file['pose'][(i)][np.newaxis])[0] # local to global

                Image.fromarray(rgb).save(f'all/rgb/{val}_{i:05d}.png')
                Image.fromarray(dep[..., 0].astype(np.uint8)).save(f'all/depth/{val}_{i:05d}.png')

                with open(f'all/pose/{val}_{i:05d}.txt', 'w') as f:
                    f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[0])))
                    f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[1])))
                    f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[2])))
                    f.write('0.0 0.0 0.0 1.0\n')

                valid_center = project(
                    center_point,
                    np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]),
                    pose,
                    (0, 256, 0, 256),
                ) & center_keep

                valid_center_invalid_color = project(
                    center_point,
                    np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]),
                    pose,
                    (128-32, 128+32, 0, 256),
                ) & center_keep

                vertex_info = vertex_info_raw.copy()
                invalid = np.unique(center_to_vertex[valid_center_invalid_color].flatten())
                vertex_info[invalid, :3] *= 0 # set RGB to 0
                vertex_info[invalid, -1] = 0 # set unobserved

                voxel_pts = center_point[valid_center]
                voxel_to_vertex = center_to_vertex[valid_center]
                required_vertex_idx, inverse = np.unique(voxel_to_vertex.flatten(), return_inverse=True)

                np.savez_compressed(
                    f'all/npz/{val}_{i:05d}.npz',
                    voxel_pts=voxel_pts.astype(np.float32),
                    voxel_to_vertex=inverse.reshape((-1, 8)).astype(np.int32),
                    vertex_idx=vertex_3d_idx[required_vertex_idx].astype(np.int32),
                    vertex_info=vertex_info[required_vertex_idx].astype(np.int32),
                    reference=reference.astype(np.float32),
                )

                np.savez_compressed(
                    f'all/target_ft/{val}_{i:05d}.npz',
                    vertex_feature=vertex_feature[required_vertex_idx].astype(np.float32),
                    vertex_feature_valid=vertex_feature[required_vertex_idx, 0].astype(np.float32) * 0 + 1,
                )


            


