import os, h5py
import numpy as np
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
    pts = pts.dot(np.linalg.inv(int_mat).T) # local
    pts = pts * np.stack([dep] * 3, axis=-1)
    pts = np.concatenate([pts, np.ones((h, w, 1), np.float32)], axis=-1)
    pts = pts.dot(ext_mat.T)[..., :3]
    return pts


if __name__ == '__main__':

    # Office 3

    os.system('mkdir office3')
    os.system('mkdir office3/pose')
    os.system('mkdir office3/depth')
    os.system('mkdir office3/rgb')
    os.system('mkdir office3/init_voxel')

    # min -5.10428727 -1.20933279 -0.62640018
    # max 1.90538916 1.52193518 5.93462044

    with open('office3/bbox.txt', 'w') as f:
        f.write('-5.11 -1.21 -0.63 1.91 1.53 5.94 0.6\n')
    # for voxel_size in [0.2, 0.4, 0.6, 0.8, 1.0]:
    #     check_bbox(f'-5.11 -1.21 -0.63 1.91 1.53 5.94 {voxel_size}')
    # quit()

    with open('office3/intrinsics.txt', 'w') as f:
        f.write('128.0 0.0 128.0 0.0\n')
        f.write('0.0 128.0 128.0 0.0\n')
        f.write('0.0 0.0 1.0 0.0\n')
        f.write('0.0 0.0 0.0 1.0\n')

    h5_filename = '/home/lzq/lzy/mono_dep_novel_view/dataset/val_replica_pose.h5'
    h5_file = h5py.File(h5_filename, 'r')

    all_pts = []

    for i, (rgb, dep, pose) in enumerate(zip(h5_file['rgb'][()], h5_file['dep'][()], pose2rotmat(h5_file['pose'][()]))):

        if i == 200:
            break

        pts = unproject(dep[..., 0] / 255.0 * 10.0, np.array([[128.0,0,128.0],[0,128.0,128.0],[0,0,1]]), pose)
        all_pts.append(pts[dep[..., 0]>0].reshape((-1, 3)))
        
        val = int(i >= 40)
        Image.fromarray(rgb).save(f'office3/rgb/{val}_{i:04d}.png')
        Image.fromarray(dep[..., 0].astype(np.uint8)).save(f'office3/depth/{val}_{i:04d}.png')
        with open(f'office3/pose/{val}_{i:04d}.txt', 'w') as f:
            f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[0])))
            f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[1])))
            f.write('%.16lf %.16lf %.16lf %.16lf\n' % tuple(list(pose[2])))
            f.write('0.0 0.0 0.0 1.0\n')
    
    all_pts = np.concatenate(all_pts, axis=0)
    vox_min = all_pts.min(axis=0)
    vox_max = all_pts.max(axis=0)
    for voxel_size in [0.2, 0.4, 0.6, 0.8, 1.0]:
        pc = np.unique(np.floor((all_pts - vox_min) / voxel_size), axis=0) * voxel_size + vox_min
        # print(voxel_size)
        # print(vox_max)
        # print(pc.max(axis=0) + voxel_size)
        # print(pc.shape[0], check_bbox(list(vox_min) + list(vox_max) + [voxel_size]))
        # print()
        with open(f'office3/init_voxel/fine_points_{voxel_size}.txt', 'w') as f:
            for pt in pc:
                f.write('%.6lf %.6lf %.6lf\n' % tuple(list(pt)))

