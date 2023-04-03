import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from net import SparseConvUNet

def to_rgb(ft):
    min_val, max_val = ft.min(axis=0), ft.max(axis=0)
    return np.round((ft - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

def collate_fn(data):
    for idx, d in enumerate(data):
        if idx == 0:
            continue
        for key in data[0]:
            if key == 'batch':
                data[0][key] = np.concatenate([data[0][key], d[key] + idx])
            else:
                data[0][key] = np.concatenate([data[0][key], d[key]])
    return {k: torch.from_numpy(data[0][k]) for k in data[0]}

class ReplicaGenDataset(Dataset):

    def __init__(self, npz_folder, sid_list=[]):
        self.files = []
        for sid in sid_list:
            self.files += sorted(glob.glob(npz_folder + f'/{sid:02d}_*.npz'))
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file, allow_pickle=True)
        data = {k: data[k] for k in data}
        data['batch'] = (data['vertex_idx'][:, :1] * 0).astype(np.int32)
        data['idx'] = np.array([idx])
        data['vertex_input'][:, -1:] *= -1
        data['vertex_input'][:, -1:] += 1.0

        fill_mask = np.round(data['vertex_input'][:, -1]).astype(np.bool)
        known_mask = ~fill_mask

        known_ft = data['vertex_input'][known_mask, :-1]
        ft_mu = known_ft.mean(axis=0)
        ft_sigma = known_ft.std(axis=0)

        data['vertex_output_norm'] = (data['vertex_output'] - ft_mu) / ft_sigma
        data['vertex_input_norm'] = data['vertex_input'] * 0
        data['vertex_input_norm'][fill_mask, -1] = 1
        data['vertex_input_norm'][known_mask, :-1] = data['vertex_output_norm'][known_mask]
        data['norm_mu'] = ft_mu
        data['norm_sigma'] = ft_sigma

        assert(np.abs(data['vertex_input_norm'][fill_mask, :-1]).mean() < 1e-6)
        assert(np.abs(data['vertex_input_norm'][known_mask, :-1] - data['vertex_output_norm'][known_mask]).mean() < 1e-6)

        if False:
            with open(f'pts/{idx}_in.txt', 'w') as f:
                for (x,y,z),(r,g,b) in zip(data['vertex_idx'], to_rgb(ft_in[:, :3])):
                    f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))
            with open(f'pts/{idx}_out.txt', 'w') as f:
                for (x,y,z),(r,g,b) in zip(data['vertex_idx'], to_rgb(data['vertex_output'][:, :3])):
                    f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))
    
        return data

def forward(model, data, loss_fn):
    batch = data['batch'].long().cuda()
    vertex_idx = data['vertex_idx'].long().cuda()
    vertex_input = data['vertex_input_norm'].float().cuda()
    vertex_output = data['vertex_output_norm'].float().cuda()
    fill_mask = torch.round(vertex_input[..., -1]).bool()

    batch_size = batch.max().item() + 1
    location = torch.cat([vertex_idx, batch], dim=-1).long().contiguous()
    pred = model((location, vertex_input, batch_size))

    print(pred.cpu().numpy().std(axis=0).mean())

    location = location.detach().cpu().numpy()
    pred_ft = pred.detach().cpu().numpy()
    pred_ft = np.concatenate([pred_ft, pred_ft[:,:1]], axis=-1)
    gt_ft = vertex_output.detach().cpu().numpy()
    gt_ft = np.concatenate([gt_ft, gt_ft[:,:1]], axis=-1)
    for i in range(batch_size):
        mask = (location[:, -1] == i)
        pts = location[mask][:, :3]
        ft = pred_ft[mask]
        gt = gt_ft[mask]
        to_vis = (ft + 3.0) / 6.0 * 255.0
        to_vis = np.round(np.clip(to_vis, 0.0, 255.0)).astype(np.uint8)
        gt_to_vis = (gt + 3.0) / 6.0 * 255.0
        gt_to_vis = np.round(np.clip(gt_to_vis, 0.0, 255.0)).astype(np.uint8)
        for ii in range(11):
            with open(f'pts/pred_{i}_{ii:02d}.txt', 'w') as f:
                for (x,y,z), (r,g,b) in zip(pts, to_vis[:,ii*3:ii*3+3]):
                    f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))
            with open(f'pts/gt_{i}_{ii:02d}.txt', 'w') as f:
                for (x,y,z), (r,g,b) in zip(pts, gt_to_vis[:,ii*3:ii*3+3]):
                    f.write('%d;%d;%d;%d;%d;%d\n' % (x,y,z,r,g,b))

    quit()
    return criterion(pred[fill_mask], vertex_output[fill_mask])


def train_model(model, data_loader, loss_fn, optimizer, ckpt_path=None, val_data_loader=None):

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.31622776601)

    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(1, 100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            optimizer.zero_grad()

            loss = forward(model, data, loss_fn)

            print(f'Epoch {epoch} Iter {it}', loss.item(), data['idx'])

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(d, f'ckpt_unet_gen_ftio/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    print(f'Val Epoch {epoch} Iter {it}', val_data['idx'])
                    with torch.no_grad():
                        loss = forward(model, data, loss_fn)
                        print(f'Val Loss {loss.item():.6f}')
            
            sys.stdout.flush()
        
        scheduler.step()

    return

def test_model(model, data_loader, loss_fn, ckpt_path):

    d = torch.load(ckpt_path)
    model.load_state_dict(d['model_state_dict'])

    with torch.no_grad():
        for data in data_loader:
            loss = forward(model, data, loss_fn)
            print(f'Test Loss {loss.item():.6f}')


if __name__ == '__main__':

    val_sid = [13,14,19,20,21,42]
    train_sid = [i for i in range(48) if i not in val_sid]

    train_data = ReplicaGenDataset('../ReplicaGenFtTriplets/easy/npz', train_sid)
    val_data = ReplicaGenDataset('../ReplicaGenFtTriplets/easy/npz', val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True, collate_fn=collate_fn)

    device = 'cuda:0'
    scn_unet = SparseConvUNet(33, 32, torch.Tensor([144, 64, 160]).long(), reps=2, n_planes=[i * 48 for i in [1, 2, 3, 4, 5]]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=3.1622776601e-5)

    # train_model(scn_unet, train_dataloader, criterion, optimizer, 'ckpt_unet_gen_ftio/ckpt_1_0.pt', val_dataloader)
    test_model(scn_unet, train_dataloader, criterion, 'ckpt_unet_gen_ftio/ckpt_4_4000.pt')
