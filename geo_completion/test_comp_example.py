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
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--load_ckpt", type=str, default="ckpt/ckpt_geo_comp.pt")
    parser.add_argument("--start_from", type=int, default=0)
    return parser.parse_args()


class ExampleScenesFused(Dataset):

    def __init__(self, voxel_size, data_dir):
        self.voxel_size = voxel_size
        self.scenes = sorted(glob.glob(f'{data_dir}/*'))
        return
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        pts = np.load(f'{self.scenes[idx]}/npz/00000.npz')['pts']
        coords_input = ME.utils.quantization.sparse_quantize(pts / self.voxel_size)
        return coords_input, idx


def collate_fn(list_data):
    coords_in, idx = list(zip(*list_data))
    return {
        'coords_in': ME.utils.batched_coordinates(coords_in),
        'idx': idx,
    }


def forward(net, data, save=None):

    device = next(net.parameters()).device

    sin = ME.SparseTensor(
        features=torch.ones(data['coords_in'].shape[0], 1),
        coordinates=data['coords_in'],
        device=device,
    )
    cm = sin.coordinate_manager

    target_keys = []

    # Generate from a dense tensor
    out_cls, targets, sout = net(sin, target_keys)
    for _ in range(10):
        sout = ME.SparseTensor(
            features=torch.ones(sout.C.shape[0], 1),
            coordinates=sout.C,
            device=device,
        )
        out_cls, targets, sout = net(sout, target_keys)

    if save is not None:
        li_in = sin.decomposed_coordinates
        li_pd = sout.decomposed_coordinates
        for idx, pc_pd in enumerate(li_pd):
            # np.savetxt(f'{save}_in.txt', li_in[idx].cpu().numpy(), fmt='%d', delimiter=' ')
            # np.savetxt(f'{save}_out.txt', pc_pd.cpu().numpy(), fmt='%d', delimiter=' ')
            np.savez_compressed(f'{save}_pred.npz', pred_geo=pc_pd.cpu().numpy())
    
    return sout


def set_bn_track_stats_false(net):
    for name, child in net.named_children():
        if 'bn' in name:
            child.track_running_stats = False
        else:
            set_bn_track_stats_false(child)
    return

def test_model(net, data_loader, args):

    ckpt = torch.load(args.load_ckpt)

    # net.eval()
    set_bn_track_stats_false(net)
    net.training = False

    data_iter = iter(data_loader)
    for it, data_dict in enumerate(tqdm.tqdm(data_iter)):
        with torch.no_grad():
            # print(f'Load checkpoint {args.load_ckpt}')
            net.load_state_dict(ckpt['model_state_dict'])
            sout = forward(net, data_dict, save=f'mink_comp_pred/{it}')
            # print(f'Test Iter {it}')
    return


if __name__ == '__main__':

    args = get_args()

    data_dir = '../ExampleScenesFused'
    voxel_size = 0.10

    test_data = ExampleScenesFused(voxel_size, data_dir)

    torch.manual_seed(1993)
    torch.cuda.manual_seed(1993)

    test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = CompletionNet(in_nchannel=1)
    net.to(device)

    test_model(net, test_data_loader, args)

