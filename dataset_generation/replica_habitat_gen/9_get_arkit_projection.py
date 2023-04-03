import numpy as np
import tqdm, random
from PIL import Image
import os, glob, torch
from utils import unproject, filter_points, project, project_depth, PseudoColorConverter

if __name__ == '__main__':

    # selects = [
    #     '41159519_38253007_38252007_38256006',
    #     '41159519_38305002_38303003_38307001',
    #     '41159566_35213620_35212621_35217619',
    #     '41159566_35355614_35354614_35358613',
    #     '42446156_79162312_79160313_79163312',
    #     '42446156_79162312_79161313_79164311',
    #     '42898497_749915075_749914076_749916075',
    #     '45663099_54885049_54884049_54886048',
    # ]

    with open('/home/lzq/lzy/NSVF/selected_arkit_views.txt') as f:
        selects = [line.strip().split() for line in f.readlines()]
    

    gt_files = [file.replace('npz_pred_valonly', 'depth') for file in sorted(glob.glob(f'/work/lzq/matterport3d/ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/npz_pred_valonly/*.npz'))]
    basenames = [os.path.basename(file).replace('.npz', '') for file in gt_files]
    pd_files = sorted(glob.glob(f'/work/lzq/matterport3d/ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/mink_nsvf_rgba_sep_field_base/output_depth/depth/*.npz'))
    idx2pd = {}
    for pd_file, gt_file in zip(pd_files, gt_files):
        idx2pd[os.path.basename(gt_file.replace('.npz', ''))] = pd_file

    int_mat = np.array([
        [211.949, 0.0, 127.933],
        [0.0,211.949,95.9333],
        [0.0,0.0,1.0],
    ])

    for select, rot in selects:
        sid, k, i, j = select.split('_')#[int(num) for num in ]


        def read(idx):
            npz = np.load(f'/home/lzq/lzy/NSVF/ARKitScenesSingleViewNSVF/all/npz/{idx}.npz')
            return npz['pts'], npz['rgb']
        
        pts_i, info_i = read(f'{sid}_{i}')
        pts_j, info_j = read(f'{sid}_{j}')
        pts = np.concatenate([pts_i, pts_j], axis=0)
        info = np.concatenate([info_i, info_j], axis=0)

        poses = np.load(f'/home/lzq/lzy/arkit_new_val_video_poses/{select}.npz')['poses']



        gt_depth = np.load(f'/home/lzq/lzy/NSVF/ARKitScenesSingleViewNSVF/all/depth/{sid}_{k}.npz')['depth']
        dep_min, dep_max = gt_depth.min(), gt_depth.max()
        converter = PseudoColorConverter('viridis', 0, dep_max)
        gt_dep_vis = converter.convert(gt_depth)

        results = []
        dep_results = []
        for pose in poses:

            out = np.zeros((192, 256, 3), np.uint8)
            dep = np.zeros((192, 256), np.float32)

            valid_mask = project(pts, int_mat, pose, valid=(0, 256, 0, 192))
            uv, d = project_depth(pts, int_mat, pose)
            uv = np.floor(uv).astype(np.int32)
            d = d[valid_mask]
            order = np.argsort(d)[::-1]
            d = d[order]
            uv = uv[valid_mask][order]
            vinfo = info[valid_mask][order]

            out[uv[:,1], uv[:,0]] = vinfo
            dep[uv[:,1], uv[:,0]] = d

            results.append(Image.fromarray(out).rotate(int(rot), expand=1))
            dep_results.append(Image.fromarray(converter.convert(dep)))

        os.system(f'mkdir -p paper_arkit_input/rgb/{select}')
        os.system(f'mkdir -p paper_arkit_input/dep/{select}')
        Image.fromarray(gt_dep_vis).save(f'paper_arkit_input/dep/{select}/gt.jpg')


        pd = np.load(idx2pd[select])['arr_0'].reshape((192, 256))
        
        Image.fromarray(converter.convert(pd)).save(f'paper_arkit_input/dep/{select}/pd.jpg')

        px = torch.load(f'/home/lzq/lzy/PixelSynth/res_multiviews_arkit/{select}/proj_depth_center_frame.pt').numpy()
        Image.fromarray(converter.convert(px)).save(f'paper_arkit_input/dep/{select}/px.jpg')

        try:
            spsg = np.load(f'../tianxing_pred_depth/video_spsg_arkit/{select}/007.npz')['depth'] * 0.1
            Image.fromarray(converter.convert(spsg)).save(f'paper_arkit_input/dep/{select}/spsg.jpg')
        except Exception as err:
            print(err)

        try:
            nf = np.load(f'../tianxing_pred_depth/video_arkit_pxnerf/{select}/center_depth.npz')['depth'].reshape((15, 192, 256))[7]
            Image.fromarray(converter.convert(nf)).save(f'paper_arkit_input/dep/{select}/nf.jpg')
        except Exception as err:
            print(err)

        for t in range(15):
            results[t].save(f'paper_arkit_input/rgb/{select}/{t:02d}.jpg')
            dep_results[t].save(f'paper_arkit_input/dep/{select}/{t:02d}.jpg')
        os.system(f'mkdir -p paper_arkit_input/gif')
        gif_li = results[7:] + results[::-1] + results[:7]
        gif_li[0].save(f'paper_arkit_input/gif/{select}.gif', save_all=True, append_images=gif_li[1:], duration=200, loop=0)

