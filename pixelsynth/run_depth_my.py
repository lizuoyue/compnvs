import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

from torchvision import utils

import os
import copy
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

from models.networks.architectures import Unet
# from options.train_options import ArgumentParser

# Generate depth for samples in validation set
# CUDA_VISIBLE_DEVICES=8 python run_depth_my.py --load_params_from models/depth/checkpoints/replica/UNet_36.pt

class ReplicaDepthDataset(Dataset):
    def __init__(self, phase):
        super(ReplicaDepthDataset, self).__init__()
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        # self.rgb_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/rgb'  # {sid:02d}_{vid:03d}.png
        self.completed_rgb_dir = 'res_complete_replica'  # {sid:02d}_{vid:03d}_{i:03d}_{j:03d}.png

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
        self.view_names = [img[:-4] for img in os.listdir(self.completed_rgb_dir) if '.png' in img]

    def __len__(self):
        return len(self.view_names)

    def __getitem__(self, i):
        view_name = self.view_names[i]
        completed_rgb = Image.open(os.path.join(self.completed_rgb_dir, f'{view_name}.png')).convert("RGB")
        completed_rgb = self.transform(completed_rgb)
        return view_name, completed_rgb


class ReplicaMidDepthDataset(Dataset):
    def __init__(self, phase):
        super(ReplicaMidDepthDataset, self).__init__()
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        # self.rgb_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/rgb'  # {sid:02d}_{vid:03d}.png
        self.completed_rgb_dir = 'res_refined_replica_mid'  # {sid:02d}_{vid:03d}_{i:03d}_{j:03d}.png

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
        self.view_names = [img[:-4] for img in os.listdir(self.completed_rgb_dir) if '.png' in img]
        # self.view_names = ['13_116_017_166', '13_171_021_101', '14_132_039_052', 
        #                    '19_025_073_213', '19_087_031_153']

    def __len__(self):
        return len(self.view_names)

    def __getitem__(self, i):
        view_name = self.view_names[i]
        completed_rgb = Image.open(os.path.join(self.completed_rgb_dir, f'{view_name}.png')).convert("RGB")
        completed_rgb = self.transform(completed_rgb)
        return view_name, completed_rgb

class ARKitDepthDataset(torch.utils.data.Dataset):
    def __init__(self, phase):
        super(ARKitDepthDataset, self).__init__()
        self.data_dir = '/home/lzq/lzy/ARKitScenes/Selected'
        self.phase = phase
        self.split_file = f'{self.data_dir}/{phase}_split.txt'
        with open(self.split_file, 'r') as txt_file:
            self.scene_names = [line.strip() for line in txt_file.readlines()]

        self.completed_rgb_dir = f'res_complete_arkit'  # {sid:02d}_{vid:03d}_{i:03d}_{j:03d}.png
        self.view_names = [img[:-4] for img in os.listdir(self.completed_rgb_dir) if '.png' in img]

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])

    def __len__(self):
        return len(self.view_names)

    def __getitem__(self, i):
        view_name = self.view_names[i]
        completed_rgb = Image.open(os.path.join(self.completed_rgb_dir, f'{view_name}.png')).convert("RGB")
        completed_rgb = self.transform(completed_rgb)
        return view_name, completed_rgb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_G", type=str, default="sync:spectral_batch")
    parser.add_argument("--dataset", type=str, choices=['arkit', 'replica', 'replica_mid'])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--load_params_from", type=str, default=None)
    args = parser.parse_args()

    depth_estimator = Unet(channels_in=3, channels_out=1, opt=args, num_filters=32).cuda()

    base_epoch = 0
    if args.load_params_from:
        if os.path.exists(args.load_params_from):
            depth_estimator.load_state_dict(torch.load(args.load_params_from))
            print(f'Loaded state_dict from {args.load_params_from}.')
            base_epoch = int(args.load_params_from[-5:-3]) + 1
        else:
            raise Exception(f'Given parameter dir {args.load_params_from} does not exist.')

    if args.dataset == 'replica':
        datasets = {phase: ReplicaDepthDataset(phase) for phase in ['val']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase=='train'),
                        num_workers=0) for phase in ['val']}
    elif args.dataset == 'arkit':
        datasets = {phase: ARKitDepthDataset(phase) for phase in ['val']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase=='train'),
                        num_workers=0) for phase in ['val']}
    elif args.dataset == 'replica_mid':
        datasets = {phase: ReplicaMidDepthDataset(phase) for phase in ['val']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase=='train'),
                        num_workers=0) for phase in ['val']}
    
    os.makedirs(f'res_depth_{args.dataset}/imgs', exist_ok=True)
    os.makedirs(f'res_depth_{args.dataset}/torch', exist_ok=True)

    for phase in ['val']:
        depth_estimator.eval()   # Set model to evaluate mode
        
        for sample_idx, samples in enumerate(tqdm(dataloaders[phase])):
            view_names = samples[0]
            rgbs = samples[1].cuda()  # Bx3x256x256
            # gt_depths = samples[2].cuda().unsqueeze(1) # Bx1x256x256

            with torch.set_grad_enabled(False):
                pred_depths = depth_estimator(rgbs) # Bx1x256x256
                if args.dataset == 'arkit':
                    pred_depths = F.interpolate(pred_depths, size=(192, 256), mode='nearest')

            bs = pred_depths.shape[0]
            for bidx in range(bs):
                view_name = view_names[bidx]
                pred_depth = pred_depths[bidx,0].detach().cpu()
                depth_save_dir = os.path.join(f'res_depth_{args.dataset}/torch', view_name+'.pt')
                torch.save(pred_depth, depth_save_dir)

                utils.save_image((pred_depth.clamp(0.5, 10) - 0.5)/9.5, 
                                  os.path.join(f'res_depth_{args.dataset}/imgs', view_name+'.png'))
