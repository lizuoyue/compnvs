import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from net import SparseConvUNet

class ReplicaGenDataset(Dataset):

    def __init__(self, npz_folder, sid_list=[]):
        self.files = []
        for sid in sid_list:
            self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_*.npz'))
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        print('Data', idx)
        file = self.files[idx]
        data = np.load(file, allow_pickle=True)
        d = {}
        d['vertex_idx'] = torch.from_numpy(data['vertex_idx']).cuda()
        d['vertex_input'] = torch.from_numpy(data['vertex_input']).cuda()
        d['vertex_output'] = torch.from_numpy(data['vertex_output']).cuda()
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

            vertex_idx = data['vertex_idx'][0]
            vertex_input = data['vertex_input'][0]
            vertex_output = data['vertex_output'][0]

            optimizer.zero_grad()
            pred = model((vertex_idx, vertex_input, 1))
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
                torch.save(d, f'ckpt_unet_new/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    print(f'Val Epoch {epoch} Iter {it}')
                    val_data = next(val_data_iter)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter)
                    val_model(model, val_data, loss_fn)
            
            sys.stdout.flush()

    return

def val_model(model, data, loss_fn):
    
    with torch.no_grad():

        vertex_idx = data['vertex_idx'][0]
        vertex_input = data['vertex_input'][0]
        vertex_output = data['vertex_output'][0]

        pred = model((vertex_idx.contiguous(), vertex_input, 1))
        loss = criterion(pred, vertex_output)

        print(f'Val Loss {loss.item():6f}')


if __name__ == '__main__':

    val_sid = [13,14,19,20,21,42]
    train_sid = [i for i in range(48) if i not in val_sid]

    train_data = ReplicaGenDataset('../ReplicaGenTriplets/all/npz', train_sid)
    val_data = ReplicaGenDataset('../ReplicaGenTriplets/all/npz', val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    device = 'cuda:0'
    scn_unet = SparseConvUNet(4, 32, torch.Tensor([256, 256, 256]).long(), reps=2, n_planes=[i * 64 for i in [1, 2, 3, 4, 5]]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=0.001)

    train_model(scn_unet, train_dataloader, criterion, optimizer, None, val_dataloader)

