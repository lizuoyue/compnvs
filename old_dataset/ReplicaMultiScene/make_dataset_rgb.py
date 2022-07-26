import os, h5py, tqdm
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

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

def check_bbox(bbox):
    # bbox = [float(num) for num in bbox.split()]
    # assert(len(bbox) == 7)
    x_min, y_min, z_min, x_max, y_max, z_max, size = bbox
    # print('X %.2lf' % ((x_max - x_min) / size))
    # print('Y %.2lf' % ((y_max - y_min) / size))
    # print('Z %.2lf' % ((z_max - z_min) / size))
    x_num = np.ceil((x_max - x_min) / size).astype(np.int32)
    y_num = np.ceil((y_max - y_min) / size).astype(np.int32)
    z_num = np.ceil((z_max - z_min) / size).astype(np.int32)
    # print()
    return x_num * y_num * z_num

def unproject(dep, int_mat, ext_mat):
    assert(len(dep.shape) == 2)
    h, w = dep.shape
    x, y = np.meshgrid(np.arange(w).astype(np.float32)+0.5, np.arange(h).astype(np.float32)+0.5)
    z = np.ones((h, w), np.float32)
    pts = np.stack([x, y, z], axis=-1)
    pts = pts.dot(np.linalg.inv(int_mat).T)
    pts = pts * np.stack([dep] * 3, axis=-1) # local
    local = pts.copy()
    pts = np.concatenate([pts, np.ones((h, w, 1), np.float32)], axis=-1)
    pts = pts.dot(ext_mat.T)[..., :3]
    return local, pts


if __name__ == '__main__':

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

    voxel_size = 0.1
    half_voxel_size = voxel_size / 2
    ckpt = torch.load(f'multi_scene_nsvf_basev1_10cm_refine/checkpoint_last.pt')
    h5_filename = '/home/lzq/lzy/mono_dep_novel_view/dataset/train_replica_pose.h5'
    h5_file = h5py.File(h5_filename, 'r')
    val = 0

    for scene_idx in range(45):

        scene_name = f'scene{scene_idx:02d}'
        center_point = ckpt['model'][f'encoder.all_voxels.{scene_idx}.points'].numpy().astype(np.float32)
        center_to_vertex = ckpt['model'][f'encoder.all_voxels.{scene_idx}.feats'].numpy().astype(np.int32)
        vertex_feature = ckpt['model'][f'encoder.all_voxels.{scene_idx}.values.weight'].numpy().astype(np.float32)
        ref = center_point.min(axis=0) - half_voxel_size
    
        all_pts, all_colors = [], []
        for i in tqdm.tqdm(list(range(scene_idx * 250, scene_idx * 250 + 250))):

            rgb = h5_file['rgb'][(i)]
            dep = h5_file['dep'][(i)]
            pose = pose2rotmat(h5_file['pose'][(i)][np.newaxis])[0]

            local, pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), pose)
            all_pts.append(pts[dep[..., 0]>0].reshape((-1, 3)))
            all_colors.append(rgb[dep[..., 0]>0].reshape((-1, 3)))
        
        all_pts = np.concatenate(all_pts, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        which_vertex_3d = np.round((all_pts - ref) / voxel_size).astype(np.int32) * 2

        center_3d_idx_raw = (center_point - ref) / half_voxel_size
        center_3d_idx = np.round(center_3d_idx_raw).astype(np.int)
        print(np.abs(center_3d_idx - center_3d_idx_raw).mean())

        vertex_3d_idx_raw = np.repeat(center_3d_idx, 8, axis=0) + np.tile(offset, (center_3d_idx.shape[0], 1))
        vertex_3d_idx = -np.ones((vertex_feature.shape[0], 3), np.int32)
        d = {}
        for (x, y, z), i in zip(vertex_3d_idx_raw, center_to_vertex.reshape((-1))):
            if vertex_3d_idx[i, 0] != -1:
                x0, y0, z0 = vertex_3d_idx[i]
                assert(x == x0)
                assert(y == y0)
                assert(z == z0)
            else:
                vertex_3d_idx[i, 0] = x
                vertex_3d_idx[i, 1] = y
                vertex_3d_idx[i, 2] = z
                d[(x, y, z)] = i
        
        which_vertex = []
        for x, y, z in which_vertex_3d:
            which_vertex.append(d[(x, y, z)])
        which_vertex = np.array(which_vertex, np.int32)

        vertex_rgbn = np.zeros((vertex_feature.shape[0], 4), np.int32)
        for color, i in zip(all_colors, which_vertex):
            vertex_rgbn[i, :3] += color.astype(np.int32)
            vertex_rgbn[i, 3] += 1

        with open(f'{scene_name}/init_voxel/vertex_{voxel_size}_with_rgb.txt', 'w') as f:
            for pt, color in zip(vertex_3d_idx, vertex_rgbn):
                assert(pt[0] > -1)
                x, y, z = pt * half_voxel_size + ref
                if color[3] > 0:
                    r, g, b = (color[:3] / color[3]).astype(np.uint8)
                else:
                    r, g, b = 0, 0, 0
                f.write('%.6lf;%.6lf;%.6lf;%d;%d;%d;%d\n' % (x, y, z, r, g, b, color[3]))

