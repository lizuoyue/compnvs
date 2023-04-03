import torch
import numpy as np
from glob import glob
from PIL import Image
import pickle as pkl
import heapq
import os, sys, tqdm, glob, random
import argparse
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import models.lmconv.get_custom_order


class ReplicaMidMaskDataset(Dataset):
    def __init__(self, phase):
        self.data_dir = '/home/lzq/lzy/PixelSynth/gen_data_pytorch3d_mid'

        self.phase = phase
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        self.views = []
        for sid in self.scene_ids:
            scene_data_dir = os.path.join(self.data_dir, f'{sid:02d}')
            scene_views = sorted(glob.glob(os.path.join(scene_data_dir, f'*_mask.jpg')))
            scene_views = [v[:-9] for v in scene_views]
            self.views += scene_views
    
    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx):
        view_dir = self.views[idx]
        view_name = '_'.join(view_dir.split('/')[-2:])  #  {sid:02d}_{k:03d}_{i:03d}_{j:03d})
        mask = np.array(Image.open(view_dir + '_mask.jpg').convert('L').resize((64, 64)))
        # rgb = np.array(Image.open(view_dir + '_in.jpg').convert('RGB').resize((64, 64)))
        return mask, view_name

class ReplicaMaskDataset(Dataset):
    def __init__(self, phase):
        self.data_dir = '/home/lzq/lzy/PixelSynth/gen_data_pytorch3d'

        self.phase = phase
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        self.views = []
        for sid in self.scene_ids:
            scene_data_dir = os.path.join(self.data_dir, f'{sid:02d}')
            scene_views = sorted(glob.glob(os.path.join(scene_data_dir, f'*_mask.jpg')))
            scene_views = [v[:-9] for v in scene_views]
            self.views += scene_views
    
    def __len__(self):
        return len(self.views)

    def __getitem__(self, idx):
        view_dir = self.views[idx]
        view_name = '_'.join(view_dir.split('/')[-2:])  #  {sid:02d}_{k:03d}_{i:03d}_{j:03d})
        mask = np.array(Image.open(view_dir + '_mask.jpg').convert('L').resize((64, 64)))
        # rgb = np.array(Image.open(view_dir + '_in.jpg').convert('RGB').resize((64, 64)))
        return mask, view_name

class ARKitMaskDataset(Dataset):
    def __init__(self, phase):
        self.data_dir = '/home/lzq/lzy/arkit_scenes_proj'

        self.phase = phase
        self.split_file = f'/home/lzq/lzy/ARKitScenes/Selected/{phase}_split.txt'
        with open(self.split_file, 'r') as txt_file:
            self.scene_names = [line.strip() for line in txt_file.readlines()]

        self.mask_dirs = []
        for scene_name in self.scene_names:
            # scene_data_dir = os.path.join(self.data_dir, f'{sid:02d}')
            scene_mask_dirs = sorted(glob.glob(os.path.join(self.data_dir, 'src', f'{scene_name}*mask.jpg')))
            self.mask_dirs += scene_mask_dirs
    
    def __len__(self):
        return len(self.mask_dirs)

    def __getitem__(self, idx):
        mask_dir = self.mask_dirs[idx]
        # mask = np.array(Image.open(mask_dir).convert('L').resize((64, 48)))
        mask = np.array(Image.open(mask_dir).convert('L').resize((64, 64)))[0:48]
        view_name = os.path.basename(mask_dir)[:-9] # remove '_mask.jpg'
        return mask, view_name


class GeneralMaskDataset(Dataset):
    def __init__(self, h, w):
        h2, w2 = h // 2, w // 2
        h4, w4 = 8, 8 #h // 4, w // 4
        self.masks = []
        #
        mask = np.zeros((h, w), np.float32)
        mask[:, :w2] = 255.0
        self.masks.append(mask)
        #
        mask = np.zeros((h, w), np.float32)
        mask[:, w2:] = 255.0
        self.masks.append(mask)
        #
        mask = np.zeros((h, w), np.float32)
        mask[:h2, :] = 255.0
        self.masks.append(mask)
        #
        mask = np.zeros((h, w), np.float32)
        mask[h2:, :] = 255.0
        self.masks.append(mask)
        #
        mask = np.zeros((h, w), np.float32)
        mask[h4:-h4, w4:-w4] = 255.0
        self.masks.append(mask)
        return
    
    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask = self.masks[idx]
        view_name = str(idx)
        return mask, view_name

class MaskFolderDataset(Dataset):
    def __init__(self, mask_folder):
        import glob
        self.mask_files = sorted(glob.glob(mask_folder + '/*g'))
        return
    
    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        # mask = Image.open(self.mask_files[idx]).resize((128, 96)).convert('RGB')
        mask = Image.open(self.mask_files[idx]).resize((64, 48)).convert('RGB')
        mask = (np.array(mask)[..., 0] > 128).astype(np.float32) * 255.0
        view_name = str(idx)
        return mask, view_name


def downsample(input):
    return F.avg_pool2d(
        input,
        kernel_size=3,
        stride=2,
        padding=[1, 1],
        count_include_pad=False,
    )

def get_custom_order_idx(rows, cols, distances, mass_center):
    # print(distances.shape, mass_center.shape)
    # distances: h x w
    # mass_center: 2 x 
    idx = []
    r, c = mass_center
    diff = c - r
    tot = mass_center[0] + mass_center[1]

    distances *= 10000

    ddd = np.max(distances)

    # c = np.argmax(distances) % rows
    c = np.argmax(distances) % cols
    # r = int((np.argmax(distances)-c) / rows)
    r = int((np.argmax(distances)-c) / cols)
    final_order = [[ddd, r, c]]
    #final_distances = []
    used = [[r, c]]
    candidate_distances = []
    #import pdb 
    #pdb.set_trace()
    while len(final_order) < rows * cols:
        # add candidates surrounding new 
        if r - 1 >= 0 and [r-1,c] not in used: # Up
            heapq.heappush(candidate_distances,(-distances[r-1,c], [r-1,c])) 
            used.append([r-1,c])
            #candidate_distances.append(distances[r-1,c])
        if r + 1 < rows and [r+1,c] not in used: # Down
            heapq.heappush(candidate_distances,(-distances[r+1,c], [r+1,c])) 
            used.append([r+1,c])
            #candidate_distances.append(distances[r+1,c])
        if c - 1 >= 0 and [r,c-1] not in used: # Left 
            heapq.heappush(candidate_distances,(-distances[r,c-1], [r,c-1])) 
            used.append([r,c-1])
            #candidate_distances.append(distances[r,c-1])
        if c + 1 < cols and [r,c+1] not in used: # Right
            heapq.heappush(candidate_distances,(-distances[r,c+1], [r,c+1])) 
            used.append([r,c+1])   
            #candidate_distances.append(distances[r,c+1])
        (neg_dist, [r,c]) = heapq.heappop(candidate_distances)
        final_order.append([-neg_dist, r, c])
    return np.array(final_order)

# def get_custom_order_idx(rows, cols, distances, mass_center):
#     return models.lmconv.get_custom_order.custom_idx(rows, cols, distances, mass_center)

# obs = [3, 24, 32] if args.dataset == 'arkit' else [3, 32, 32]   # ch, rows (H), cols (W)
def get_masks_for_batch(background_mask, obs):
    # 1 if in foreground, 0 if no points nearby
    foreground_mask = ~background_mask

    # downsample mask to be autoregressive size, convert bool to float
    background_mask = downsample(background_mask.float())
    foreground_mask = downsample(foreground_mask.float())
    # print(foreground_mask.max(), foreground_mask.min())

    b, h, w = background_mask.shape

    # multiply by index to get center of mass
    y=torch.arange(h).view(1,h,1)
    x=torch.arange(w).view(1,1,w)
    
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    mass_x = foreground_mask * x
    mass_y = foreground_mask * y

    mass_center_x = torch.mean(mass_x.view(b,-1),axis=1).cpu().detach()
    mass_center_y = torch.mean(mass_y.view(b,-1),axis=1).cpu().detach()
    mass_center = torch.stack((mass_center_x, mass_center_y), axis=1).numpy().astype(int)

    # get distance of each pixel to nearest background pixel
    # and distance of background pixels to nearest non-background pixel
    bin_fg_mask = foreground_mask.view(b, h, w, 1).cpu().detach().numpy().astype(np.uint8)
    bin_bg_mask = background_mask.view(b, h, w, 1).cpu().detach().numpy().astype(np.uint8)
    foreground_distances = np.zeros((b,h,w))
    background_distances = np.zeros((b,h,w))
    for image_num in range(b):
        foreground_distances[image_num] = cv2.distanceTransform(bin_fg_mask[image_num], distanceType=cv2.DIST_L2, maskSize=5)
        background_distances[image_num] = cv2.distanceTransform(bin_bg_mask[image_num], distanceType=cv2.DIST_L2, maskSize=5)
    distances = (foreground_distances - background_distances).astype(int)

    # import matplotlib; matplotlib.use('agg')
    # import matplotlib.pyplot as plt
    # plt.imshow(distances[0])
    # plt.colorbar()
    # plt.savefig('aaaaa.png')
    # plt.clf()
    # input('see fig')
    # print(distances.max(), distances.min())
    #example = np.array([[0,1,1,1],[1,1,1,1],[1,1,0,0],[0,0,0,0]]).astype(np.uint8)

    # autoregressive algorithm is as follows:
    # begin from maximum distance to background pixels
    # and proceed towards background pixels; fill in these
    # closest to foreground pixels, then furtherst
    # ties are broken using spiral pattern, 
    # which starts from center of mass.
    gen_orders = []
    # masks_init = []
    # masks_undilated = []
    # masks_dilated = []
    
    for image_num in range(b):
        gen_order = get_custom_order_idx(obs[1], obs[2], distances[image_num], mass_center[image_num])
        gen_orders.append(gen_order)
    #     gen_orders.append(get_generation_order_idx('custom', obs[1], obs[2], distances[image_num], mass_center[image_num]))
    #     mask_init, mask_undilated, mask_dilated = get_masks(gen_orders[-1], obs[1], obs[2], 3, 2, plot=False)#True, out_dir='log/mp3d/end_to_end_3x_upgraded_adjacent')
    #     masks_init.append(mask_init[0:1])
    #     masks_undilated.append(mask_undilated[0:1])
    #     masks_dilated.append(mask_dilated[0:1])

    # masks_init = torch.stack(masks_init).repeat(1, 513, 1, 1).view(-1,9,obs[1]*obs[2]).cuda(non_blocking=True)
    # masks_undilated = torch.stack(masks_undilated).repeat(1, 160, 1, 1).view(-1,9,obs[1]*obs[2]).cuda(non_blocking=True)
    # masks_dilated = torch.stack(masks_dilated).repeat(1, 80, 1, 1).view(-1,9,obs[1]*obs[2]).cuda(non_blocking=True)
        
    # return masks_init, masks_undilated, masks_dilated, gen_orders
    return gen_orders

def run(data_loader, phase, args):

    shape_dict = {
        'arkit': [3, 24, 32],
        'replica': [3, 32, 32],
        'replica_mid': [3, 32, 32],
        'fountain': [3, 48, 64],
        'westminster': [3, 48, 64],
        'notre': [3, 48, 64],
        'sacre': [3, 48, 64],
        'pantheon': [3, 48, 64],
    }

    with torch.no_grad():
        gen_order = {}

        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            masks, view_names = batch
            batch_size = masks.shape[0]

            bg_masks = masks < 128

            # print(bg_masks.shape)
            # from PIL import Image
            # tmp = (bg_masks.cpu().int().numpy() * 255).astype(np.uint8)
            # Image.fromarray(tmp[0]).save('aaa.png')
            # quit()

            obs = shape_dict[args.dataset] # ch, rows (H), cols (W)
            gen_orders = get_masks_for_batch(bg_masks.cuda(), obs)

            for bidx in range(batch_size):
                key = view_names[bidx]
                gen_order[key] = gen_orders[bidx]

                # print(gen_orders[bidx].shape)
                # print(gen_orders[bidx].min(axis=0))
                # print(gen_orders[bidx].max(axis=0))
                # tosee = np.zeros((32, 32), np.float32)
                # for dist, r, c in gen_orders[bidx]:
                #     tosee[r, c] = dist
                # import matplotlib; matplotlib.use('agg')
                # import matplotlib.pyplot as plt
                # plt.imshow(tosee)
                # plt.colorbar()
                # plt.savefig('aaaaa.png')
                # plt.clf()
                # plt.imshow(tosee >= 0)
                # plt.colorbar()
                # plt.savefig('bbbbb.png')
                # plt.clf()
                # input('see')
                # nnn = (~bg_masks).cpu().numpy().sum()
                # rrr = gen_orders[bidx][nnn:nnn+500]
                # ggg = gen_orders[bidx][nnn+500:nnn+1000]
                # bbb = gen_orders[bidx][nnn+1000:nnn+1500]
                # tosee[rrr[:,0], rrr[:,1]] = np.array([255,0,0])
                # tosee[ggg[:,0], ggg[:,1]] = np.array([0,255,0])
                # tosee[bbb[:,0], bbb[:,1]] = np.array([0,0,255])
                # from PIL import Image
                # Image.fromarray(tosee).save('aaa.png')
                # input('press')
        
        with open('data/%s_%s_gen_order_pytorch3d.pkl' % (args.dataset, phase), 'wb') as f:
            pkl.dump(gen_order, f)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['replica', 'replica_mid', 'arkit', 'fountain', 'westminster', 'notre', 'sacre', 'pantheon'])
    parser.add_argument("--mask_folder", type=str, default=None)
    args = parser.parse_args()

    if args.dataset == 'replica':
        val_dataset = ReplicaMaskDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        run(val_dataloader, 'val', args)

        train_dataset = ReplicaMaskDataset('train')
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        run(train_dataloader, 'train', args)
    elif args.dataset == 'arkit':
        val_dataset = ARKitMaskDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        run(val_dataloader, 'val', args)

        train_dataset = ARKitMaskDataset('train')
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        run(train_dataloader, 'train', args)
    elif args.dataset == 'replica_mid':
        val_dataset = ReplicaMidMaskDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        run(val_dataloader, 'val', args)
    
    elif args.dataset in ['fountain', 'westminster', 'notre', 'sacre', 'pantheon']:
        test_dataset = MaskFolderDataset(args.mask_folder)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        run(test_dataloader, 'test', args)

    else:
        pass