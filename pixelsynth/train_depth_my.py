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
import glob
import argparse

from models.networks.architectures import Unet
# from options.train_options import ArgumentParser

class ReplicaDepthDataset(Dataset):
    def __init__(self, phase):
        super(ReplicaDepthDataset, self).__init__()
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        self.depth_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/depth'  # {sid:02d}_{vid:03d}.npz
        self.rgb_dir = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenFt/all/rgb'  # {sid:02d}_{vid:03d}.png

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
        self.view_names = []
        for sid in self.scene_ids:
            for vid in range(300):
                view_name = f'{sid:02d}_{vid:03d}'
                self.view_names.append(view_name)

    def __len__(self):
        return len(self.view_names)

    def __getitem__(self, i):
        view_name = self.view_names[i]
        depth = torch.from_numpy(np.load(os.path.join(self.depth_dir, f'{view_name}.npz'))['depth']).unsqueeze(0)
        rgb = Image.open(os.path.join(self.rgb_dir, f'{view_name}.png')).convert("RGB")
        rgb = self.transform(rgb)
        return rgb, depth

class ARKitDepthDataset(Dataset):
    def __init__(self, phase):
        super(ARKitDepthDataset, self).__init__()
        self.data_dir = '/home/lzq/lzy/ARKitScenes/Selected'

        self.phase = phase
        self.split_file = f'{self.data_dir}/{phase}_split.txt'
        with open(self.split_file, 'r') as txt_file:
            self.scene_names = [line.strip() for line in txt_file.readlines()]

        self.depth_dir = f'{self.data_dir}/depth'  # {sid}_{vid}.npz
        self.rgb_dir = f'{self.data_dir}/rgb'  # {sid}_{vid}.png

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])

        self.view_names = []
        for scene_name in self.scene_names:
            scene_rgb_dirs = glob.glob(os.path.join(self.rgb_dir, f'{scene_name}*.png'))
            self.view_names += [os.path.basename(rgb_dir)[:-4] for rgb_dir in scene_rgb_dirs]
            # scene_rgbs = os.listdir(self.rgb_dir)

    def __len__(self):
        return len(self.view_names)

    def __getitem__(self, i):
        view_name = self.view_names[i]
        depth = torch.from_numpy(np.load(os.path.join(self.depth_dir, f'{view_name}.npz'))['depth'])
        depth = F.interpolate(depth.view(1,1,192,256), size=(256, 256), mode='nearest').squeeze(0)
        rgb = Image.open(os.path.join(self.rgb_dir, f'{view_name}.png')).convert("RGB")
        rgb = self.transform(rgb)
        return rgb, depth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['replica', 'arkit'])
    parser.add_argument("--norm_G", type=str, default="sync:spectral_batch")
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--schedule_step", type=int, default=20)
    parser.add_argument("--load_params_from", type=str, default=None)
    args = parser.parse_args()

    depth_estimator = Unet(channels_in=3, channels_out=1, opt=args, num_filters=32).cuda()
    optimizer = torch.optim.SGD(depth_estimator.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.schedule_step, gamma=0.1) 

    base_epoch = 0
    if args.load_params_from:
        if os.path.exists(args.load_params_from):
            depth_estimator.load_state_dict(torch.load(args.load_params_from))
            print(f'Loaded state_dict from {args.load_params_from}.')
            base_epoch = int(args.load_params_from[-5:-3]) + 1
        else:
            raise Exception(f'Given parameter dir {args.load_params_from} does not exist.')

    criterion = nn.L1Loss()

    if args.dataset == 'replica':
        datasets = {phase: ReplicaDepthDataset(phase) for phase in ['train', 'val']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase=='train'),
                        num_workers=0) for phase in ['train', 'val']}
    elif args.dataset == 'arkit':
        datasets = {phase: ARKitDepthDataset(phase) for phase in ['train', 'val']}
        dataloaders = {phase: DataLoader(datasets[phase], batch_size=args.batch_size, shuffle=(phase=='train'),
                        num_workers=0) for phase in ['train', 'val']}

    best_loss = float('inf')
    best_loss_epoch = 0
    best_model_wts = copy.deepcopy(depth_estimator.state_dict())
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                depth_estimator.train()  # Set model to training mode
            else:
                depth_estimator.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            for sample_idx, samples in enumerate(tqdm(dataloaders[phase])):
                rgbs = samples[0].cuda()  # Bx3x256x256
                gt_depths = samples[1].cuda() # Bx1x256x256

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred_depths = depth_estimator(rgbs)
                    # print(pred_depths.shape, gt_depths.shape)
                    loss = criterion(pred_depths, gt_depths)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if phase == 'val' and sample_idx == 0:
                    img_save_dir = f'models/depth/{args.dataset}/'
                    os.makedirs(img_save_dir, exist_ok=True)
                    utils.save_image(rgbs, img_save_dir+'_rgb.png',
                                     nrow=8, padding=5, pad_value=1)
                    utils.save_image((pred_depths.clamp(0.5, 10) - 0.5)/9.5, 
                                     img_save_dir+f'{epoch:02d}_pred.png',
                                     nrow=8, padding=5, pad_value=1)
                    utils.save_image((gt_depths.clamp(0.5, 10) - 0.5)/9.5, 
                                     img_save_dir+f'{epoch:02d}_gt.png',
                                     nrow=8, padding=5, pad_value=1)
                
                running_loss += loss.item() * rgbs.shape[0]
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}.')

            # deep copy the model
            if phase == 'val':
                if epoch_loss <= best_loss:
                    best_loss = epoch_loss
                    best_loss_epoch = epoch
                    best_model_wts = copy.deepcopy(depth_estimator.state_dict())
                    ckpt_save_file = f'models/depth/checkpoints/{args.dataset}'
                    os.makedirs(ckpt_save_file, exist_ok=True)
                    ckpt_save_name = f'UNet_{epoch+base_epoch:02d}.pt'
                    torch.save(best_model_wts, os.path.join(ckpt_save_file, ckpt_save_name))
                print(f'Best until now: {best_loss:.4f} ({best_loss_epoch}-th epoch).')
            
        scheduler.step()

    # pred_depths = (
    #                 F.sigmoid(depth_estimator(gen_img)) * (opts.max_z - opts.min_z) + opts.min_z
    #             )
