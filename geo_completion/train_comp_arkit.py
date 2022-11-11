import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys, glob, tqdm, argparse
import MinkowskiEngine as ME
from minkowski_completion_network import CompletionNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=30)
    # parser.add_argument("--val_freq", type=int, default=1000)
    # parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--num_workers", type=int, default=1)
    # parser.add_argument("--stat_freq", type=int, default=50)
    # parser.add_argument("--weights", type=str, default="modelnet_completion.pth")
    parser.add_argument("--load_ckpt", type=str, default="ckpt_comp_arkit/ckpt_4_1500.pt")
    parser.add_argument("--start_from", type=int, default=0)
    # parser.add_argument("--eval", action="store_true")
    # parser.add_argument("--max_visualization", type=int, default=4)
    return parser.parse_args()

def project(pts, int_mat, ext_mat, valid=(0, 256, 0, 256)):
    assert(len(pts.shape) == 2) # (N, 3)
    local = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=-1)
    local = local.dot(np.linalg.inv(ext_mat).T)[:, :3]
    local = local.dot(int_mat.T)
    local[:, :2] /= local[:, 2:]
    x_min, x_max, y_min, y_max = valid
    valid = np.ones((pts.shape[0],), np.bool)
    valid = valid & (local[:, 0] >= x_min)
    valid = valid & (local[:, 0] <= x_max)
    valid = valid & (local[:, 1] >= y_min)
    valid = valid & (local[:, 1] <= y_max)
    valid = valid & (local[:, 2] > 0)
    return valid

class ReplicaGeoCompDataset(Dataset):

    def __init__(self, voxel_size, triplets_folder, npz_folder, scene_ids):
        self.voxel_size = voxel_size
        self.triplets = []
        for sid in scene_ids:
            triplets = np.load(f'{triplets_folder}/{sid}.npz')['triplets']
            triplets = np.concatenate([np.ones((triplets.shape[0], 1), np.int32) * sid, triplets], axis=1)
            self.triplets.append(triplets)
        self.triplets = np.concatenate(self.triplets, axis=0)
        self.npz_folder = npz_folder
        # self.pose_folder = npz_folder.replace('npz', 'pose')

        # self.scene_pts = []
        # for sid in range(48):
        #     self.scene_pts.append(np.loadtxt(f'../replica_habitat_gen/ReplicaGen/scene{sid:02d}/init_voxel/center_points.txt'))
        # self.int_mat = np.array([
        #     [128.0,   0.0, 128.0],
        #     [  0.0, 128.0, 128.0],
        #     [  0.0,   0.0,   1.0],
        # ])
        return
    
    def __len__(self):
        return self.triplets.shape[0]

    def __getitem__(self, idx):
        sid, k, i, j = self.triplets[idx]
        pts_i = self.load_npz(sid, i) / self.voxel_size
        pts_j = self.load_npz(sid, j) / self.voxel_size
        pts_k = self.load_npz(sid, k) / self.voxel_size

        # pose_i = np.loadtxt(f'{self.pose_folder}/{sid:02d}_{i:03d}.txt')
        # pose_j = np.loadtxt(f'{self.pose_folder}/{sid:02d}_{j:03d}.txt')
        # pose_k = np.loadtxt(f'{self.pose_folder}/{sid:02d}_{k:03d}.txt')

        # pts_output = []
        # for pose in [pose_i, pose_j, pose_k]:
        #     pts_output.append(self.scene_pts[sid][project(self.scene_pts[sid], self.int_mat, pose)])
        # pts_output = np.concatenate(pts_output, axis=0) / self.voxel_size

        pts_input = np.concatenate([pts_i, pts_j], axis=0)
        pts_output = np.concatenate([pts_input, pts_k], axis=0)
        np.random.shuffle(pts_output) # only shuffled along the first axis

        coords_input = ME.utils.sparse_quantize(pts_input)
        coords_output = ME.utils.sparse_quantize(pts_output)

        return (coords_input, coords_output, self.triplets[idx])
    
    def load_npz(self, sid, fid):
        return np.load(f'{self.npz_folder}/{sid}_{fid}.npz')['pts']


def collate_fn(list_data):
    coords_in, coords_out, idx = list(zip(*list_data))
    return {
        'coords_in': ME.utils.batched_coordinates(coords_in),
        'coords_out': ME.utils.batched_coordinates(coords_out),
        'idx': np.stack(idx)
    }

def compute_iou(gt, pd):
    pr = gt.features_at_coordinates(pd.C.float()).mean()
    rc = pd.features_at_coordinates(gt.C.float()).mean()
    iou = 1.0 / (1.0 / pr + 1.0 / rc - 1.0)
    return pr.item(), rc.item(), iou.item()

def forward(net, data, loss_fn, save=None):

    device = next(net.parameters()).device

    sin = ME.SparseTensor(
        features=torch.ones(data['coords_in'].shape[0], 1),
        coordinates=data['coords_in'],
        device=device,
    )
    cm = sin.coordinate_manager
    sout_gt = ME.SparseTensor(
        features=torch.ones(data['coords_out'].shape[0], 1),
        coordinates=data['coords_out'],
        coordinate_manager=cm,
        device=device,
    )

    sin_at_gt = sin.features_at_coordinates(sout_gt.C.float())
    addtion = sout_gt.C[sin_at_gt[:, 0] < 0.5] # exist in gt but not sin

    # Generate target sparse tensor
    target_gen_key, _ = cm.insert_and_map(addtion, string_id='target_gen')
    target_roi_key, _ = cm.insert_and_map(data['coords_out'].to(device), string_id='target_roi')
    target_keys = [target_gen_key, target_roi_key]

    # Generate from a dense tensor
    out_cls, targets, sout = net(sin, target_keys)
    num_layers, loss = len(out_cls), 0
    losses = []
    # weight * (A - 1) + 1 (0 -> 1, 1 -> A)
    for out_cl, weight, target in zip(out_cls, targets[0], targets[1]):
        # print(out_cl.F.shape, target.shape, target.float().mean(), weight.shape, weight.float().mean())
        loss_before_reduce = loss_fn(out_cl.F.squeeze(), target.type(out_cl.F.dtype).to(device))
        loss_weight = weight.detach().float() * (1.0 - 1.0) + 1.0
        loss_weight /= loss_weight.sum()
        curr_loss = (loss_before_reduce * loss_weight).sum()
        losses.append(curr_loss.item())
        loss += curr_loss / num_layers
    
    sout_pd = ME.SparseTensor(
        features=torch.ones(sout.C.shape[0], 1),
        coordinates=sout.C,
        coordinate_manager=cm,
        device=device,
    )

    if save is not None:
        li_in = sin.decomposed_coordinates
        li_gt = sout_gt.decomposed_coordinates
        li_pd = sout.decomposed_coordinates
        for idx, pc_pd in enumerate(li_pd):
            sid, k, i, j = data['idx'][idx]
            np.savetxt(f'{save}/{sid:02d}_{k:03d}_{i:03d}_{j:03d}_in.txt', li_in[idx].cpu().numpy(), fmt='%d', delimiter=' ')
            np.savetxt(f'{save}/{sid:02d}_{k:03d}_{i:03d}_{j:03d}_gt.txt', li_gt[idx].cpu().numpy(), fmt='%d', delimiter=' ')
            np.savetxt(f'{save}/{sid:02d}_{k:03d}_{i:03d}_{j:03d}.txt', pc_pd.cpu().numpy(), fmt='%d', delimiter=' ')
            quit()
            np.savez_compressed(f'{save}/{sid}_{k}_{i}_{j}.npz', pred_geo=pc_pd.cpu().numpy())
    
    return loss, losses, compute_iou(sout_gt, sout_pd), sout


def train_model(net, train_data_loader, val_data_loader, args):

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    val_iter = iter(val_data_loader)
    loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    if args.load_ckpt != 'none':
        ckpt = torch.load(args.load_ckpt)
        print(f'Load checkpoint {args.load_ckpt}')
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch, _ = ckpt['epoch_it']

    net.train()
    
    for epoch in range(start_epoch, args.epoch):
        train_iter = iter(train_data_loader)
        for it, data_dict in enumerate(tqdm.tqdm(train_iter)):

            optimizer.zero_grad()
            loss, losses, ious, sout = forward(net, data_dict, loss_fn)
            print(f'Train Epoch {epoch} Iter {it}', end=' ')
            print(losses, ious)
            loss.backward()
            optimizer.step()

            if it % 500 == 0:# and it != 0:
                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(d, f'ckpt_comp_arkit/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                print(f'Val Epoch {epoch} Iter {it}', end=' ')
                data_dict = next(val_iter, None)
                if data_dict is None:
                    val_iter = iter(val_data_loader)
                    data_dict = next(val_iter, None)
                
                with torch.no_grad():
                    loss, losses, ious, sout = forward(net, data_dict, loss_fn)
                    print(losses, ious)
            
            sys.stdout.flush()

def set_bn_track_stats_false(net):
    for name, child in net.named_children():
        if 'bn' in name:
            child.track_running_stats = False
        else:
            set_bn_track_stats_false(child)
    return

def test_model(net, data_loader, args):

    ckpt = torch.load(args.load_ckpt)
    print(f'Load checkpoint {args.load_ckpt}')
    net.load_state_dict(ckpt['model_state_dict'])

    # net.eval()
    set_bn_track_stats_false(net)
    net.training = False

    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    data_iter = iter(data_loader)
    for it, data_dict in enumerate(tqdm.tqdm(data_iter)):
        with torch.no_grad():
            loss, losses, ious, sout = forward(net, data_dict, loss_fn, save=f'mink_comp_pred_arkit111')
            print(f'Test Iter {it}', end=' ')
            print(losses, ious)
            # input()
    
    return


if __name__ == '__main__':

    args = get_args()

    train_sid = np.loadtxt('/work/lzq/matterport3d/ARKitScenes/Selected/train_split.txt').astype(np.int32)
    val_sid = np.loadtxt('/work/lzq/matterport3d/ARKitScenes/Selected/val_split.txt').astype(np.int32)
    triplets_folder = '/work/lzq/matterport3d/ARKitScenes/Selected/triplets'
    npz_folder = '/work/lzq/matterport3d/ARKitScenes/ARKitScenes/ARKitScenesSingleViewNSVF/all/npz'
    voxel_size = 0.1

    train_data = ReplicaGeoCompDataset(voxel_size, triplets_folder, npz_folder, train_sid)#[args.start_from:args.start_from+300])
    val_data = ReplicaGeoCompDataset(voxel_size, triplets_folder, npz_folder, val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    torch.manual_seed(1993)
    torch.cuda.manual_seed(1993)

    train_data_loader = DataLoader(train_data, batch_size=8, shuffle=False, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = CompletionNet(in_nchannel=1)
    net.to(device)

    # train_model(net, train_data_loader, val_data_loader, args)
    # test_model(net, val_data_loader, args)
    test_model(net, train_data_loader, args)

