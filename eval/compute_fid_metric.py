import numpy as np
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
import os, glob, tqdm
import torch
import time

if __name__ == '__main__':

    gt_files, pd_files = [], []
    for sid in [13, 14, 19, 20, 21, 42]:
        # gt_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/rgb/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_complete/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/07.png'))
        pass
    # pd_files = sorted(glob.glob('/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base0302/output/color/*.png'))
    # gt_files = sorted(glob.glob(f'/home/lzq/lzy/replica_habitat_gen/replica_val_video_gt/images/train/*.png'))

    # gt_files = sorted(glob.glob(f'/home/lzq/lzy/replica_habitat_gen/replica_val_video_gt/images/train/*.png'))
    pd_files = sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output_no2d_single/color/*.png'))

    # for gt_file, pd_file in zip(gt_files, pd_files):
    #     # assert(os.path.basename(gt_file) == os.path.basename(pd_file))
    #     assert(os.path.basename(gt_file).replace('.png', '') == pd_file.split('/')[-2])
    #     gt = np.array(Image.open(gt_file))[...,:3]
    #     pd = np.array(Image.open(pd_file))[...,:3]
    #     Image.fromarray(np.hstack([gt, pd])).save('temp.png')
    #     input('press')

    dims = 2048
    batch_size = 200
    num_workers = 0
    device = torch.device('cuda:0')
    
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    # m1, s1 = fid_score.calculate_activation_statistics(gt_files, model, batch_size, dims, device, num_workers)
    # np.savez_compressed('fid/replica_gt_video.npz', m=m1, s=s1)
    npz = np.load('fid/replica_gt_center_frame.npz')
    m1, s1 = npz['m'], npz['s']
    m2, s2 = fid_score.calculate_activation_statistics(pd_files, model, batch_size, dims, device, num_workers)
    np.savez_compressed('fid/replica_no2d_center_frame.npz', m=m2, s=s2)

    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)
    print(fid_value)
