import numpy as np
from PIL import Image
import os, glob, tqdm
import torch, cv2
import multiprocessing

"""Peak Signal to Noise Ratio
img1 and img2 have range [0, 255]"""
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * np.log10(255.0 / np.sqrt(mse))

"""Structure Similarity
img1, img2: [0, 255]"""
def ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:  # Grey or Y-channel image
        return _ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(_ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return _ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")

def _ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

if __name__ == '__main__':

    rebuttal_select = [
        # '13_018_102_289',
        # '13_116_017_166',
        # '13_125_030_061',
        '13_171_021_101',
        # '13_215_004_265',
        # '13_263_034_262',
        # '13_276_014_170',
        # '14_034_054_170',
        # '14_132_039_052',
        # '14_216_026_128',
        '19_025_073_213',
        # '19_087_031_153',
        # '19_258_079_256',
        # '20_044_099_120',
        # '20_107_001_005',
    ]
    rid = 0

    gt_files, pd_files, mask_files = [], [], []
    for sid in [13, 14, 19, 20, 21, 42]:
        # gt_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/rgb/{sid:02d}_*.png'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_complete/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/07.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/*.png'))
        # mask_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/gen_data_pytorch3d_mid/{sid:02d}/*_mask.jpg'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/*.png'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_refined_replica_mid/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews_replica_mid/{sid:02d}_*/00.jpg'))
        # temp = sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/mink_nsvf_rgba_sep_field_base/outputvideo{sid:02d}/color/*.png'))
        # pd_files += [pd_file for idx, pd_file in enumerate(temp) if (idx//15)%3==0]
        pass

    pd_files = sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews_replica_mid/*/*.jpg'))
    gt_files = sorted(glob.glob(f'/home/lzq/lzy/replica_habitat_gen/replica_video_gt_midval/images/train/*.png'))

    print(len(gt_files), len(pd_files))
    # assert(len(gt_files) == len(pd_files))# == len(mask_files))

    lp, ls = [], []
    if True:
        def func(x):
            gt_file, pd_file = x#, _ = x
            gt = np.array(Image.open(gt_file))[...,:3]
            pd = np.array(Image.open(pd_file))[...,:3]
            return ssim(gt, pd)

        # def func(x):
        #     gt_file, pd_file, mask_file = x
        #     gt = np.array(Image.open(gt_file))[...,:3]
        #     pd = np.array(Image.open(pd_file))[...,:3]
        #     mask = np.array(Image.open(mask_file)) > 127
        #     pd[mask] = gt[mask]
        #     return (ssim(gt, pd) - mask.mean()) / (1-mask.mean())
                
        # def func(x):
        #     gt_file, pd_file, mask_file = x
        #     gt = np.array(Image.open(gt_file))[...,:3]
        #     pd = np.array(Image.open(pd_file))[...,:3]
        #     mask = np.array(Image.open(mask_file)) <= 127
        #     return psnr(gt[mask], pd[mask])

        li = list(zip(gt_files, pd_files)) # mask_files
        with multiprocessing.Pool(20) as p:
        #     # ls = p.map(func, list(zip(gt_files, pd_files))[:10])
        #     # print(np.mean(ls))
            res = list(tqdm.tqdm(p.imap(func, li), total=len(li)))
            print(np.mean(res))
            
        # with Pool(processes=4) as p:
        #     with tqdm(total=len(extract_list)) as pbar:
        #         for i, _ in enumerate(p.imap_unordered(extract_video_frames, extract_list)):
        #             pbar.update()

        # for gt_file, pd_file in tqdm.tqdm(list(zip(gt_files, pd_files))[26700:]):
        #     gt = np.array(Image.open(gt_file))[...,:3]
        #     pd = np.array(Image.open(pd_file))[...,:3]
        #     Image.fromarray(np.hstack([gt, pd])).save('tst.png')
        #     input()
        #     # lp.append(psnr(gt, pd))
        #     ls.append(ssim(gt, pd))
        #     # print(np.mean(lp))#, np.mean(ls))
        #     print(np.mean(ls))
    else:
        for gt_file, pd_file, mask_file in tqdm.tqdm(list(zip(gt_files, pd_files, mask_files))):

            if rebuttal_select[rid] in gt_file:

                gt = np.array(Image.open(gt_file))[...,:3]
                pd = np.array(Image.open(pd_file))[...,:3]
                # mask = np.array(Image.open(mask_file)) <= 127
                ps = np.array(Image.open(f'PixelSynth/res_refined_replica_mid/{rebuttal_select[rid]}.png'))[...,:3]
                # gt[~mask] = 0
                # pd[~mask] = 0
                # lp.append(psnr(gt[mask], pd[mask]))
                num = int(os.path.basename(pd_file).replace('.png', '')) * 15
                ims = [Image.open(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/mink_nsvf_rgba_sep_field_base/outputvideo{rebuttal_select[rid][:2]}/color/{idx:05d}.png') for idx in range(num, num+15)]

                for idx in range(15):
                    ims[idx].save(f'{rebuttal_select[rid]}/{idx:02d}.jpg')

                # ims[0].save(f'{rebuttal_select[rid]}.gif',save_all=True,append_images=ims[1:],duration=200,loop=0)
                # lp.append(psnr(gt, pd))
                # ls.append(ssim(gt, pd))
                # print(gt_file)
                # Image.fromarray(np.hstack([gt, pd, ps])).save(f'{rebuttal_select[rid]}.png')

                # print(np.mean(lp), np.mean(ls))
                # input()
                rid += 1
    

    
