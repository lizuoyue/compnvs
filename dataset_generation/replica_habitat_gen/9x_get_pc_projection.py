import numpy as np
import tqdm, random
from PIL import Image
import os, glob, torch
from utils import unproject, filter_points, project, project_depth, PseudoColorConverter

if __name__ == '__main__':

    # selects = [
    #     '13_118_090_268',
    #     '13_009_028_268',
    #     '13_032_028_090',
    #     '19_114_006_198',
    #     '19_041_198_222',
    #     '19_004_006_222',
    # ]
    selects = [
        '13_171_021_101',
        '19_025_073_213',
    ]

    gt_files = []
    for sid in [13, 14, 19, 20, 21, 42]:
        gt_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/depth/{sid:02d}_*.npz'))
    pd_files = sorted(glob.glob('/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output_no2d_single_depth/depth/*.npz'))
    idx2pd = {}
    for pd_file, gt_file in zip(pd_files, gt_files):
        idx2pd[os.path.basename(gt_file.replace('.npz', ''))] = pd_file

    int_mat = np.array([
        [128.0,   0.0, 128.0],
        [  0.0, 128.0, 128.0],
        [  0.0,   0.0,   1.0],
    ])
    bboxes = np.loadtxt('ReplicaGen/bboxes.txt')
    for select in selects:
        sid, k, i, j = [int(num) for num in select.split('_')]
        bbox = bboxes[sid]

        def read(i):
            rgb = np.array(Image.open(f'ReplicaGen/scene{sid:02d}/rgb/{i:03d}.png'))[..., :3]
            dep = np.load(f'ReplicaGen/scene{sid:02d}/depth/{i:03d}.npz')['depth']
            pose = np.loadtxt(f'ReplicaGen/scene{sid:02d}/pose/{i:03d}.txt')
            pts = unproject(dep, int_mat, pose)
            pts, mask = filter_points(pts, bbox, init_mask=dep>1e-3, return_mask=True)
            info = np.array(rgb)[..., :3][mask]#.astype(np.float32)
            return pts, info
        
        pts_i, info_i = read(i)
        pts_j, info_j = read(j)
        pts = np.concatenate([pts_i, pts_j], axis=0)
        info = np.concatenate([info_i, info_j], axis=0)

        gt_depth = np.load(f'ReplicaGen/scene{sid:02d}/depth/{k:03d}.npz')['depth']
        dep_min, dep_max = gt_depth.min(), gt_depth.max()
        converter = PseudoColorConverter('viridis', 0, dep_max)
        gt_dep_vis = converter.convert(gt_depth)

        poses = np.load(f'ReplicaGenEncFtTriplets/mid/pose_video/{select}.npz')['poses']
        results = []
        dep_results = []
        for pose in poses:

            out = np.zeros((256, 256, 3), np.uint8)

            dep = np.zeros((256, 256), np.float32)

            valid_mask = project(pts, int_mat, pose)
            uv, d = project_depth(pts, int_mat, pose)
            uv = np.floor(uv).astype(np.int32)
            d = d[valid_mask]
            order = np.argsort(d)[::-1]
            d = d[order]
            uv = uv[valid_mask][order]
            vinfo = info[valid_mask][order]

            out[uv[:,1], uv[:,0]] = vinfo
            dep[uv[:,1], uv[:,0]] = d

            results.append(Image.fromarray(out))
            dep_results.append(Image.fromarray(converter.convert(dep)))
        
        os.system(f'mkdir -p rebuttal_replica_input/rgb/{select}')
        os.system(f'mkdir -p rebuttal_replica_input/dep/{select}')
        Image.fromarray(gt_dep_vis).save(f'rebuttal_replica_input/dep/{select}/gt.jpg')

        # import cv2
        # pd = np.load(idx2pd[select])['arr_0'].reshape((256, 256))
        # if sid in [19,20,21]:
        #     pd[gt_depth < 0.1] = 0

        # pd = cv2.inpaint(pd,(np.abs(pd-5.0)<0.001).astype(np.uint8),10,cv2.INPAINT_NS)*0.83
        
        # Image.fromarray(converter.convert(pd)).save(f'rebuttal_replica_input/dep/{select}/pd.jpg')

        for t in range(15):
            results[t].save(f'rebuttal_replica_input/rgb/{select}/{t:02d}.jpg')
            dep_results[t].save(f'rebuttal_replica_input/dep/{select}/{t:02d}.jpg')
        os.system(f'mkdir -p rebuttal_replica_input/gif')
        gif_li = results[7:] + results[::-1] + results[:7]
        gif_li[0].save(f'rebuttal_replica_input/gif/{select}.gif', save_all=True, append_images=gif_li[1:], duration=200, loop=0)

