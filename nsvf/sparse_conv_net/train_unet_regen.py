import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from net import SparseConvUNet

def collate_fn(data):
    for idx, d in enumerate(data):
        if idx == 0:
            continue
        for key in data[0]:
            if key == 'batch':
                data[0][key] = torch.cat([data[0][key], d[key] + idx])
            else:
                data[0][key] = torch.cat([data[0][key], d[key]])
    return data[0]

class ReplicaGenDataset(Dataset):

    def __init__(self, npz_folder, sid_list=[], tmp=[]):
        self.files = []
        for sid in sid_list:
            self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_*.npz'))
        if len(tmp) > 0:
            for sid in tmp:
                self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_000.npz') * 300)
                # self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_001.npz') * 100)
                # self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_002.npz') * 100)
        return

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file, allow_pickle=True)
        d = {}
        d['batch'] = torch.from_numpy(data['vertex_idx'][:,:1] * 0).cuda()
        d['vertex_idx'] = torch.from_numpy(data['vertex_idx']).cuda()
        d['vertex_input'] = torch.from_numpy(data['vertex_input']).cuda()
        d['vertex_output'] = torch.from_numpy(data['vertex_output']).cuda()

        vis_min = data['vertex_output'][:, :3].min(axis=0)
        vis_max = data['vertex_output'][:, :3].max(axis=0)
        out_vis = np.round((data['vertex_output'][:, :3] - vis_min) / (vis_max - vis_min) * 255).astype(np.uint8)

        if False:
            with open(f'pts_in.txt', 'w') as f:
                for pt, (r, g, b, _) in zip(data['vertex_idx'], data['vertex_input']):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
            with open(f'pts_out.txt', 'w') as f:
                for pt, (r, g, b) in zip(data['vertex_idx'], out_vis):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
            quit()

        return d




def train_model(model, data_loader, loss_fn, optimizer, ckpt_path=None, val_data_loader=None):
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            batch = data['batch']
            vertex_idx = data['vertex_idx']
            vertex_input = data['vertex_input']
            vertex_output = data['vertex_output']

            batch_size = batch.max().item() + 1
            location = torch.cat([vertex_idx, batch], dim=-1).long().contiguous()

            optimizer.zero_grad()
            pred = model((location, vertex_input, batch_size))
            loss = criterion(pred, vertex_output)

            print(f'Epoch {epoch} Iter {it}', loss.item())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'ckpt_unet_regen/ckpt_10val_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    print(f'Val Epoch {epoch} Iter {it}')
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    val_model(model, val_data, loss_fn)
            
            sys.stdout.flush()

    return

def val_model(model, data, loss_fn):
    
    with torch.no_grad():

        batch = data['batch']
        vertex_idx = data['vertex_idx']
        vertex_input = data['vertex_input']
        vertex_output = data['vertex_output']

        batch_size = batch.max().item() + 1
        location = torch.cat([vertex_idx, batch], dim=-1).long().contiguous()

        pred = model((location, vertex_input, batch_size))
        loss = criterion(pred, vertex_output)

        print(f'Val Loss {loss.item():6f}')


if __name__ == '__main__':

    val_sid = [13,14,19,20,21,42]
    train_sid = [i for i in range(48) if i not in val_sid]

    train_data = ReplicaGenDataset('../ReplicaRegenTriplets/all/npz', train_sid, val_sid)
    val_data = ReplicaGenDataset('../ReplicaRegenTriplets/all/npz', val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=16, shuffle=True, collate_fn=collate_fn)

    device = 'cuda:0'
    scn_unet = SparseConvUNet(4, 32, torch.Tensor([256, 256, 256]).long(), reps=1, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=0.001)

    train_model(scn_unet, train_dataloader, criterion, optimizer, 'ckpt_unet_regen/ckpt_2_0.pt', val_dataloader)

