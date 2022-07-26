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

    ckpt = torch.load(f'multi_scene_nsvf_basev1_10cm_refine/checkpoint_13_63000.pt')
    h5_filename = '/home/lzq/lzy/mono_dep_novel_view/dataset/train_replica_pose.h5'
    h5_file = h5py.File(h5_filename, 'r')
    val = 0

    for scene_idx in range(45):

        scene_name = f'scene{scene_idx:02d}'
        os.system(f'mkdir {scene_name}')
        os.system(f'mkdir {scene_name}/pose')
        os.system(f'mkdir {scene_name}/depth')
        os.system(f'mkdir {scene_name}/rgb')
        os.system(f'mkdir {scene_name}/init_voxel')
        os.system(f'mkdir {scene_name}/subset')

        center_point = ckpt['model'][f'encoder.all_voxels.{scene_idx}.points'].numpy().astype(np.float32)
        center_ref = center_point.min(axis=0)

        with open(f'{scene_name}/intrinsics.txt', 'w') as f:
            f.write('128.0 0.0 128.0 0.0\n')
            f.write('0.0 128.0 128.0 0.0\n')
            f.write('0.0 0.0 1.0 0.0\n')
            f.write('0.0 0.0 0.0 1.0\n')
    
        all_pts, all_pts_and_vertex = [], []
        for i in tqdm.tqdm(list(range(scene_idx * 250, (scene_idx + 1) * 250))):

            rgb = h5_file['rgb'][(i)]
            dep = h5_file['dep'][(i)]
            pose = pose2rotmat(h5_file['pose'][(i)][np.newaxis])[0]

            local, pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), pose)
            all_pts.append(pts[dep[..., 0]>0].reshape((-1, 3)))
            # all_pts_and_vertex.append(all_pts[-1])

            if False:
                voxel_size = 0.1
                local = local[dep[..., 0]>0].reshape((-1, 3))
                local = np.unique(np.floor(local / voxel_size), axis=0) * voxel_size + voxel_size * 0.5
                local = np.repeat(local, 8, axis=0) + np.tile(offset * voxel_size * 0.5, (local.shape[0], 1))
                local = np.unique(local, axis=0)
                local = np.concatenate([local, np.ones((local.shape[0], 1), np.float32)], axis=-1)
                local = local.dot(pose.T)[..., :3]
                all_pts_and_vertex.append(local)

                # with open(f'test1.txt', 'w') as f:
                #     for pt in all_pts_and_vertex[-2]:
                #         f.write('%.6lf;%.6lf;%.6lf\n' % tuple(list(pt)))
                # with open(f'test2.txt', 'w') as f:
                #     for pt in all_pts_and_vertex[-1]:
                #         f.write('%.6lf;%.6lf;%.6lf\n' % tuple(list(pt)))
                
                # input()

            # Image.fromarray(rgb).save(f'{scene_name}/rgb/{val}_{i:04d}.png')
            # Image.fromarray(dep[..., 0].astype(np.uint8)).save(f'{scene_name}/depth/{val}_{i:04d}.png')
            # with open(f'{scene_name}/pose/{val}_{i:04d}.txt', 'w') as f:
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[0])))
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[1])))
            #     f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[2])))
            #     f.write('0.0 0.0 0.0 1.0\n')

        view_split = [item.shape[0] for item in all_pts]
        all_pts = np.concatenate(all_pts, axis=0)
        # all_pts_and_vertex = np.concatenate(all_pts_and_vertex, axis=0)
        # vox_min = all_pts.min(axis=0)
        # vox_max = all_pts.max(axis=0)
        # with open(f'{scene_name}/bbox.txt', 'w') as f:
        #     f.write('-5.11 -1.21 -0.63 1.91 1.53 5.94 0.6\n')
        # with open(f'{scene_name}/min_max.txt', 'w') as f:
        #     f.write('%.6lf %.6lf %.6lf\n' % tuple(list(vox_min)))
        #     f.write('%.6lf %.6lf %.6lf\n' % tuple(list(vox_max)))

        for voxel_size in [0.1]:# [0.2, 0.4]:#, 0.6, 0.8, 1.0]:
            ref = center_ref - voxel_size * 0.5
            pc, inv = np.unique(np.floor((all_pts - ref) / voxel_size).astype(np.int), axis=0, return_inverse=True)
            view_split = torch.from_numpy(inv).split(view_split)
            pc = pc * voxel_size + center_ref
            with open(f'{scene_name}/init_voxel/fine_points_{voxel_size}.txt', 'w') as f:
                for pt in pc:
                    f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))
            for view_idx, pc_subset in enumerate(view_split):
                with open(f'{scene_name}/subset/0_{view_idx:04d}.txt', 'w') as f:
                    for num in np.unique(pc_subset.numpy()):
                        f.write('%d\n' % num)
            break
            ref = center_ref - voxel_size * 0.5
            pc = np.unique(np.floor((all_pts - ref) / voxel_size), axis=0) * voxel_size + center_ref
            with open(f'{scene_name}/init_voxel/fine_points_{voxel_size}.txt', 'w') as f:
                for pt in pc:
                    f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))
            pc = np.unique(np.floor(all_pts / voxel_size), axis=0) * voxel_size + voxel_size * 0.5
            with open(f'{scene_name}/init_voxel/fine_points_{voxel_size}_aligned.txt', 'w') as f:
                for pt in pc:
                    f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))
            pc = np.unique(np.floor(all_pts_and_vertex / voxel_size), axis=0) * voxel_size + voxel_size * 0.5
            with open(f'{scene_name}/init_voxel/fine_points_{voxel_size}_with_vertex.txt', 'w') as f:
                for pt in pc:
                    f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))

