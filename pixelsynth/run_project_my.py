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

from models.networks.architectures import ResNetDecoder # refinement model
from models.vqvae2.vqvae import VQVAETop

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import compositing
from pytorch3d.renderer.points import rasterize_points


INT_MAT_REPLICA = torch.from_numpy(np.array([
    [128.0,   0.0, 128.0],
    [  0.0, 128.0, 128.0],
    [  0.0,   0.0,   1.0],
])).float().cuda()
IMG_SIZE = 256
INT_MAT_ARKIT = torch.from_numpy(np.array([
    [211.949, 0.0, 127.933],
    [0.0, 211.949, 95.9333],
    [0.0, 0.0, 1.0],
])).float().cuda()

def project_depth(pts, int_mat, ext_mat):
    assert(len(pts.shape) == 2) # (N, 3)
    local = torch.cat([pts, torch.ones(pts.shape[0], 1).float().cuda()], dim=-1)
    local = local.matmul(torch.linalg.inv(ext_mat).transpose(0, 1))[:, :3]
    local = local.matmul(int_mat.transpose(0, 1))
    local[:, :2] /= local[:, 2:]
    return local[:, :2], local[:, 2]

def pytorch3d_project(coords, colors, poses, dataset):
    if dataset == 'replica' or dataset == 'replica_mid':
        int_mat = INT_MAT_REPLICA
    elif dataset == 'arkit':
        int_mat = INT_MAT_ARKIT
    imgs = []
    for pose in poses:
        uv, d = project_depth(coords, int_mat, pose)
        uv = (uv / -IMG_SIZE) * 2.0 + 1.0

        pts = torch.stack([uv[:,0], uv[:,1], d], dim=1).float()

        fts = torch.cat([colors, d.unsqueeze(1)], dim=-1)

        radius = 4.0 / float(IMG_SIZE) * 2.0

        pts3D = Pointclouds(points=[pts], features=[fts])
        points_idx, _, dist = rasterize_points(
            pts3D, (IMG_SIZE, IMG_SIZE), radius, 8
        )

        alphas = (1 - dist.clamp(max=1, min=1e-3).pow(0.5)).permute(0, 3, 1, 2)

        results = compositing.alpha_composite(
            points_idx.permute(0, 3, 1, 2).long(),
            alphas,
            pts3D.features_packed().permute(1,0),
        )

        no_proj = (points_idx.permute(0, 3, 1, 2).long() == -1).int().sum(dim=1, keepdim=True) == 8
        no_proj = torch.cat([no_proj]*4,dim=1)
        results[no_proj] = -1

        if dataset == 'arkit':
            imgs.append(results[:,:,:192])
        elif dataset == 'replica' or dataset == 'replica_mid':
            imgs.append(results)

    return torch.cat(imgs, dim=0)


def unproject(dep, int_mat, ext_mat):
    # int_mat: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # ext_mat: local to world
    assert(len(dep.shape) == 2)
    h, w = dep.shape
    y, x = torch.meshgrid(torch.arange(h).cuda().float()+0.5, torch.arange(w).float().cuda()+0.5)
    z = torch.ones(h, w).float().cuda()
    pts = torch.stack([x, y, z], dim=-1)
    pts = pts.matmul(torch.linalg.inv(int_mat[:3, :3]).transpose(0, 1))
    pts = pts * torch.stack([dep] * 3, dim=-1) # local
    pts = torch.cat([pts, torch.ones(h, w, 1).float().cuda()], dim=-1)
    pts = pts.matmul(ext_mat.transpose(0, 1))[..., :3]
    return pts

def process_fn(pred_img, pred_dep, mask, in_coord, in_color, k_pose, poses, dataset):
    if dataset == 'replica' or dataset == 'replica_mid':
        int_mat = INT_MAT_REPLICA
    elif dataset == 'arkit':
        int_mat = INT_MAT_ARKIT
    k_coords = unproject(pred_dep, int_mat, k_pose)
    pred_coords = k_coords[mask.bool()]
    pred_colors = pred_img.permute([1,2,0])[mask]
    coord = torch.cat([in_coord, pred_coords], dim=0)
    color = torch.cat([in_color, pred_colors], dim=0)
    return pytorch3d_project(coord, color, poses, dataset)

class ReplicaResultsDataset(Dataset):
    def __init__(self, phase):
        super(ReplicaResultsDataset, self).__init__()
        assert phase == 'val'
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        self.depth_res_dir = 'res_depth_replica/torch'  # {sid:02d}_{vid:03d}.pt
        self.complete_imgs_dir = 'res_complete_replica'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.outpaint_imgs_dir = 'res_outpaint_replica'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.mask_dir = 'gen_data_pytorch3d'    # {sid:02d}/{k:03d}_{i:03d}_{j:03d}_mask.jpg
        self.data_src_dir = '/home/lzq/lzy/replica_habitat_gen'

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
        self.sample_names = sorted([img[:-4] for img in os.listdir(self.complete_imgs_dir) if '.png' in img])
        # self.sample_names = [name for name in self.sample_names if int(name[:2]) in [19, 20, 21]]

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        sid, k, i, j = [int(item) for item in sample_name.split('_')]
        # view_name = sample_name[:6]
        depth = torch.load(os.path.join(self.depth_res_dir, f'{sample_name}.pt'))
        complete_img = self.transform(Image.open(os.path.join(self.complete_imgs_dir, f'{sample_name}.png')))
        outpaint_img = self.transform(Image.open(os.path.join(self.outpaint_imgs_dir, f'{sample_name}.png')))
        mask = torch.from_numpy(np.array(Image.open(os.path.join(self.mask_dir, f'{sid:02d}/{k:03d}_{i:03d}_{j:03d}_mask.jpg')))) <= 127

        # if sid in [19,20,21]:
        #     # k_rgb = np.array(Image.open(f'{self.data_src_dir}/ReplicaGenFt/all/rgb/{sid:02d}_{k:03d}.png'))[...,:3]
        #     black = complete_img < -0.999999
        #     black = black[0] & black[1] & black[2]
        #     mask[black] = False
        mask[depth <= 1e-4] = False
        mask[depth >= 10] = False

        k_pose = torch.from_numpy(np.loadtxt(f'{self.data_src_dir}/ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt')).float()
        
        i_data = np.load(f'{self.data_src_dir}/ReplicaGenFt/all/npz/{sid:02d}_{i:03d}.npz')
        j_data = np.load(f'{self.data_src_dir}/ReplicaGenFt/all/npz/{sid:02d}_{j:03d}.npz')
        in_coord = torch.from_numpy(np.concatenate([i_data['pts'], j_data['pts']], axis=0)).float()
        in_color = torch.from_numpy(np.concatenate([i_data['rgb'], j_data['rgb']], axis=0)).float() / 127.5 - 1.0

        poses = torch.from_numpy(np.load(f'{self.data_src_dir}/ReplicaGenEncFtTriplets/easy/pose_video/{sample_name}.npz')['poses'])

        return sample_name, complete_img, outpaint_img, depth, mask, in_coord, in_color, k_pose, poses

class ArkitResultsDataset(Dataset):
    def __init__(self, phase):
        super(ArkitResultsDataset, self).__init__()
        assert phase == 'val'
        import glob
        self.pose_dir = '/home/lzq/lzy/arkit_new_val_video_poses'
        self.sample_names = sorted(glob.glob(f'{self.pose_dir}/*.npz'))
        self.sample_names = [os.path.basename(file).replace('.npz', '') for file in self.sample_names]

        self.depth_res_dir = 'res_depth_arkit/torch'  # {sid:02d}_{vid:03d}.pt
        self.complete_imgs_dir = 'res_complete_arkit'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.outpaint_imgs_dir = 'res_outpaint_arkit'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.mask_dir = '/home/lzq/lzy/arkit_scenes_proj/src'    # {sid:02d}_{k:03d}_{i:03d}_{j:03d}_mask.jpg
        self.data_src_dir = '../ARKitScenes/Selected'

        self.transform = transforms.Compose([
            transforms.Resize((192, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        sid, k, i, j = sample_name.split('_')
        view_name = f'{sid}_{k}'
        depth = torch.load(os.path.join(self.depth_res_dir, f'{sample_name}.pt'))
        complete_img = self.transform(Image.open(os.path.join(self.complete_imgs_dir, f'{sample_name}.png')))
        outpaint_img = self.transform(Image.open(os.path.join(self.outpaint_imgs_dir, f'{sample_name}.png')))
        mask = torch.from_numpy(np.array(Image.open(os.path.join(self.mask_dir, f'{sample_name}_mask.jpg')).convert('L'))) <= 127
        mask = mask[0:192]

        k_pose = torch.from_numpy(np.loadtxt(f'{self.data_src_dir}/pose/{view_name}.txt')).float()
        
        i_data = np.load(f'{self.data_src_dir}/npz/{sid}_{i}.npz')
        j_data = np.load(f'{self.data_src_dir}/npz/{sid}_{j}.npz')
        in_coord = torch.from_numpy(np.concatenate([i_data['pts'], j_data['pts']], axis=0)).float()
        in_color = torch.from_numpy(np.concatenate([i_data['rgb'], j_data['rgb']], axis=0)).float() / 127.5 - 1.0

        poses = torch.from_numpy(np.load(f'{self.pose_dir}/{sample_name}.npz')['poses']).float()

        return sample_name, complete_img, outpaint_img, depth, mask, in_coord, in_color, k_pose, poses

class ReplicaMidResultsDataset(Dataset):
    def __init__(self, phase):
        super(ReplicaMidResultsDataset, self).__init__()
        assert phase == 'val'
        if phase == 'val':
            self.scene_ids = [13, 14, 19, 20, 21, 42]
        elif phase == 'train':
            self.scene_ids = [i for i in range(48) if i not in [13, 14, 19, 20, 21, 42]]

        self.depth_res_dir = 'res_depth_replica_mid/torch'  # {sid:02d}_{vid:03d}.pt
        self.complete_imgs_dir = 'res_refined_replica_mid'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.outpaint_imgs_dir = 'res_outpaint_replica_mid'  # {sid:02d}_{k:03d}_{i:03d}_{j:03d}.png
        self.mask_dir = 'gen_data_pytorch3d_mid'    # {sid:02d}/{k:03d}_{i:03d}_{j:03d}_mask.jpg
        self.data_src_dir = '/home/lzq/lzy/replica_habitat_gen'

        self.transform = transforms.Compose([
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                        ])
        self.sample_names = sorted([f[:-3] for f in os.listdir(self.depth_res_dir) if '.pt' in f])

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        sample_name = self.sample_names[idx]
        sid, k, i, j = [int(item) for item in sample_name.split('_')]
        # view_name = sample_name[:6]
        depth = torch.load(os.path.join(self.depth_res_dir, f'{sample_name}.pt'))
        complete_img = self.transform(Image.open(os.path.join(self.complete_imgs_dir, f'{sample_name}.png')))
        # outpaint_img = self.transform(Image.open(os.path.join(self.outpaint_imgs_dir, f'{sample_name}.png')))
        outpaint_img = torch.empty((1,))
        mask = torch.from_numpy(np.array(Image.open(os.path.join(self.mask_dir, f'{sid:02d}/{k:03d}_{i:03d}_{j:03d}_mask.jpg')))) <= 127

        # if sid in [19,20,21]:
        #     # k_rgb = np.array(Image.open(f'{self.data_src_dir}/ReplicaGenFt/all/rgb/{sid:02d}_{k:03d}.png'))[...,:3]
        #     black = complete_img < -0.999999
        #     black = black[0] & black[1] & black[2]
        #     mask[black] = False
        mask[depth <= 1e-4] = False
        mask[depth >= 10] = False

        k_pose = torch.from_numpy(np.loadtxt(f'{self.data_src_dir}/ReplicaGenFt/all/pose/{sid:02d}_{k:03d}.txt')).float()
        
        i_data = np.load(f'{self.data_src_dir}/ReplicaGenFt/all/npz/{sid:02d}_{i:03d}.npz')
        j_data = np.load(f'{self.data_src_dir}/ReplicaGenFt/all/npz/{sid:02d}_{j:03d}.npz')
        in_coord = torch.from_numpy(np.concatenate([i_data['pts'], j_data['pts']], axis=0)).float()
        in_color = torch.from_numpy(np.concatenate([i_data['rgb'], j_data['rgb']], axis=0)).float() / 127.5 - 1.0

        poses = torch.from_numpy(np.load(f'{self.data_src_dir}/ReplicaGenEncFtTriplets/mid/pose_video/{sample_name}.npz')['poses']) # 15x4x4

        return sample_name, complete_img, outpaint_img, depth, mask, in_coord, in_color, k_pose, poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['arkit', 'replica', 'replica_mid'])
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--load_params_from", type=str, default=None)
    # refinement model
    parser.add_argument("--predict_residual", action="store_true", default=False)
    parser.add_argument("--refine_model_type", type=str, default="resnet_256W8UpDown3")
    parser.add_argument("--norm_G", type=str, default="sync:spectral_batch")
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--normalize_before_residual", action="store_true", default=False)
    parser.add_argument("--losses", type=str, nargs="+", default=['1.0_l1','10.0_content'])
    parser.add_argument("--discriminator_losses", type=str, default="pix2pixHD")
    parser.add_argument("--lr_d", type=float, default=1e-3 * 2)
    parser.add_argument("--lr_g", type=float, default=1e-3 / 2)
    parser.add_argument("--beta1", type=float, default=0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--norm_D", type=str, default="spectralinstance")
    parser.add_argument("--gan_mode", type=str, default="hinge", help="(ls|original|hinge)")
    parser.add_argument("--no_ganFeat_loss", action="store_true")
    parser.add_argument("--lambda_feat", type=float, default=10.0)
    args = parser.parse_args()

    refine_model = ResNetDecoder(args, channels_in=3, channels_out=3).cuda()
    if args.dataset == 'replica':
        load_params_from = 'models/lmconv_with_refine/runs/replica/0_ep27.pth'
        vqvae_ckpt = torch.load('models/vqvae2/checkpoint/replica/vqvae_150.pt')
        vqvae_ckpt = {k[7:]:v for k, v in vqvae_ckpt.items()}
        val_dataset = ReplicaResultsDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    elif args.dataset == 'arkit':
        load_params_from = 'models/lmconv_with_refine/runs/arkit/0_ep16.pth'
        vqvae_ckpt = torch.load('models/vqvae2/checkpoint/arkit/vqvae_100.pt')
        vqvae_ckpt = {k[7:]:v for k, v in vqvae_ckpt.items()}
        val_dataset = ArkitResultsDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if args.dataset == 'replica_mid':
        load_params_from = 'models/lmconv_with_refine/runs/replica/0_ep27.pth'
        vqvae_ckpt = torch.load('models/vqvae2/checkpoint/replica/vqvae_150.pt')
        vqvae_ckpt = {k[7:]:v for k, v in vqvae_ckpt.items()}
        val_dataset = ReplicaMidResultsDataset('val')
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    refine_model.eval()   # Set model to evaluate mode
    refine_model.load_state_dict(torch.load(load_params_from)['refine_model_state_dict'])
    
    for sample_idx, samples in enumerate(tqdm(val_dataloader)):
        sample_names, complete_imgs, outpaint_imgs, depths, masks, in_coord, in_color, k_pose, poses = samples
        bs = complete_imgs.shape[0]

        complete_imgs, outpaint_imgs, depths, masks = complete_imgs.cuda(), outpaint_imgs.cuda(), depths.cuda(), masks.cuda()
        in_coord, in_color, k_pose, poses = in_coord.cuda(), in_color.cuda(), k_pose.cuda(), poses.cuda()

        for bidx in range(bs):

            # TODO: implement projector, no batch
            multiviews = process_fn(complete_imgs[bidx], depths[bidx], masks[bidx], in_coord[bidx], 
                                    in_color[bidx], k_pose[bidx], poses[bidx], args.dataset)    # 15x3x256x256
            
            if False:
                res = ((multiviews[:,:3]+1.0)*127.5).cpu().numpy()
                res = np.round(res).astype(np.uint8).transpose([0, 2, 3, 1])
                res = [Image.fromarray(frame) for frame in res]
                res[0].save('test.gif', duration=250, loop=0, save_all=True, append_images=res[1:])

                # sample_name = sample_names[bidx]
                # multiviews_save_dir = os.path.join(f'res_multiviews_{args.dataset}_proj_depth', sample_name)
                # os.makedirs(multiviews_save_dir, exist_ok=True)

                # import matplotlib; matplotlib.use('agg')
                # import matplotlib.pyplot as plt
                # plt.imshow(multiviews[0, -1].cpu().numpy())
                # plt.savefig('tst.png')
                # plt.clf()
                # input()
                # np.savez_compressed(f'{multiviews_save_dir}/07.npz', multiviews[0, -1].cpu().numpy())
                # continue
            
            num_views = multiviews.shape[0]
            # assert num_views == 15

            # proj_depth = multiviews[:,3]
            # multiviews = multiviews[:,:3]
            multiviews, proj_depth = torch.split(multiviews, (3, 1), dim=1)
            # print(f'multiviews: {multiviews.shape}, proj_depth: {proj_depth.shape}.')
            
            sample_name = sample_names[bidx]
            multiviews_save_dir = os.path.join(f'res_multiviews_{args.dataset}', sample_name)
            os.makedirs(multiviews_save_dir, exist_ok=True)

            # Refine the projected frames and save each
            with torch.set_grad_enabled(False):
                refined_view = refine_model(multiviews, None)
                for view_idx in range(num_views):
                    utils.save_image(refined_view[view_idx]*.5+.5, os.path.join(multiviews_save_dir, f'{view_idx:02d}.jpg'))
            # # Save GIF
            # ims = [Image.open(os.path.join(multiviews_save_dir, f'{view_idx:02d}.jpg')) for view_idx in range(num_views)]
            # ims[0].save(os.path.join(multiviews_save_dir, f'gif.gif'), append_images=ims[1:], duration=200, loop=0, save_all=True)
            # # Save the projected depth of the center frame
            # torch.save(proj_depth, os.path.join(multiviews_save_dir, 'proj_depth.pt'))