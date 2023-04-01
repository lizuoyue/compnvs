import numpy as np
from PIL import Image
import os, glob, tqdm
import torch
import lpips, sys

def transform(img):
    return (img / 127.5) - 1.0

if __name__ == '__main__':

    loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
    # loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization
    # loss_fn_sq = lpips.LPIPS(net='squeeze').cuda()

    gt_files, pd_files, mask_files = [], [], []
    for sid in [13, 14, 19, 20, 21, 42]:
        # gt_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/rgb/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_complete/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/07.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/*.png'))
        # gt_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/rgb/{sid:02d}_*.png'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_complete/{sid:02d}_*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/07.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews/{sid:02d}_*/*.png'))
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/*.png'))
        # mask_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/gen_data_pytorch3d_mid/{sid:02d}/*_mask.jpg'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/*.png'))[::3]
        # pd_files += sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews_replica_mid/{sid:02d}_*/00.jpg'))

        # temp = sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/mid/mink_nsvf_rgba_sep_field_base/outputvideo{sid:02d}/color/*.png'))
        # pd_files += [pd_file for idx, pd_file in enumerate(temp) if (idx//15)%3==0]
        pass

    pd_files = sorted(glob.glob(f'/home/lzq/lzy/PixelSynth/res_multiviews_replica_mid/*/*.jpg'))
    gt_files = sorted(glob.glob(f'/home/lzq/lzy/replica_habitat_gen/replica_video_gt_midval/images/train/*.png'))

    # gt_files = sorted(glob.glob(f'/home/lzq/lzy/replica_habitat_gen/replica_val_video_gt/images/train/*.png'))

    # pd_files = sorted(glob.glob('/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base0302/output/color/*.png'))
    # pd_files = sorted(glob.glob(f'/home/lzq/lzy/NSVF/ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output_no2d_single/color/*.png'))
    print(len(gt_files), len(pd_files))
    assert(len(gt_files) == len(pd_files))

    # a = int(sys.argv[1])
    
    alex, vgg, sq = [], [], []
    li = []
    # for gt_file, pd_file, mask_file in tqdm.tqdm(list(zip(gt_files, pd_files, mask_files))):#[a*13500:a*13500+13500]):
    for gt_file, pd_file in tqdm.tqdm(list(zip(gt_files, pd_files))):
        # assert(os.path.basename(gt_file) == os.path.basename(pd_file))
        gt = transform(torch.from_numpy(np.array(Image.open(gt_file))[...,:3].transpose([2,0,1])[np.newaxis]).cuda())
        pd = transform(torch.from_numpy(np.array(Image.open(pd_file))[...,:3].transpose([2,0,1])[np.newaxis]).cuda())
        # mask = torch.from_numpy(np.array(Image.open(mask_file)) > 127)
        # pd[:,:,mask] = gt[:,:,mask]

        alex.append(loss_fn_alex(gt, pd).item())# / (1-mask.float().mean()))
        # vgg.append(loss_fn_vgg(gt, pd).item())
        # sq.append(loss_fn_sq(gt, pd).item())

        # li.append((loss_fn_alex(gt, pd).item(), gt_file, pd_file, mask_file))

        print(np.mean(alex), np.mean(vgg), np.mean(sq))

        # if not pd_file.startswith('/home/lzq/lzy/PixelSynth/res_multiviews/42_241_126_289'):
        #     continue
        # gt = np.array(Image.open(gt_file))[...,:3]
        # pd = np.array(Image.open(pd_file))[...,:3]
        # Image.fromarray(np.hstack([gt, pd])).save('tst.png')
        # input('press')
    
    quit()
    
    li.sort()

    for _, gt_file, pd_file, mask_file in li:
        print(gt_file)
        gt = np.array(Image.open(gt_file))[...,:3]
        pd = np.array(Image.open(pd_file))[...,:3]
        Image.fromarray(np.hstack([gt, pd])).save('tst.png')
        input('press')
        