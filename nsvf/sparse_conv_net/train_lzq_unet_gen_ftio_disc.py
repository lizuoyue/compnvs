import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, glob, tqdm, os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from net import SparseConvUNet
import sparseconvnet as scn

# From CycleGan
class Discriminator(nn.Module):
    def __init__(self, num_ft_channels):
        super(Discriminator, self).__init__()

        channels = num_ft_channels
        dim = 3

        def discriminator_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [scn.Convolution(dim, in_filters, out_filters, 2, 2, False)]
            layers.append(scn.BatchNormLeakyReLU(out_filters, leakiness=0.2))
            return layers

        self.model = nn.Sequential(
            scn.InputLayer(dim, torch.Tensor([144, 64, 160]).long(), mode=4),
            *discriminator_block(channels, 64),
            *discriminator_block(64, 96),
            *discriminator_block(96, 128), 
            scn.Convolution(dim, 128, 1, 2, 2, False),
            # scn.OutputLayer(dim)
        )

    def forward(self, feat):
        output = self.model(feat).features
        return nn.functional.sigmoid(output)

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
        data = dict(data)   # ['center_pts', 'center_to_vertex', 'vertex_idx', 'vertex_input', 'vertex_output', 'reference']
        data['batch'] = (data['vertex_idx'][:, :1] * 0).astype(np.int32)
        data['idx'] = np.array([idx])
        return data

def train_model(data_loader, 
                model , loss_fn, optimizer, 
                disc, disc_loss_fn, disc_optimizer, 
                ckpt_path=None, val_data_loader=None):
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])

    ckpt_savefile = 'ckpt_lzq_unet_gen_ftio_disc2'
    ckpt_savefile += npz_folder.split('/')[-1].split('_')[-1]
    os.makedirs(ckpt_savefile, exist_ok=True)
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            batch = data['batch'].long().cuda()
            vertex_idx = data['vertex_idx'].long().cuda()
            vertex_input = data['vertex_input'].float().cuda()
            vertex_output = data['vertex_output'].float().cuda()
            mask = (1 - vertex_input[..., -1]).bool()

            # print(f'vertex_output.shape: {vertex_output.shape}.')
            num_ft, num_ft_channels = vertex_output.shape

            batch_size = batch.max().item() + 1
            location = torch.cat([vertex_idx, batch], dim=-1).long().contiguous()
            pred = model((location, vertex_input, batch_size))

            ######### Generator Phase #########
            optimizer.zero_grad()
            gen_loss = criterion(pred, vertex_output)
            gen_loss_mask = criterion(pred[mask], vertex_output[mask])

            # Loss measures generator's ability to fool the discriminator
            gt_disc_out = disc((location, vertex_output, batch_size))
            valid = torch.ones_like(gt_disc_out, requires_grad=False).cuda()
            fake = torch.zeros_like(gt_disc_out, requires_grad=False).cuda()
            gen_disc_loss = disc_loss_fn(disc((location, pred, batch_size)), valid)  # disc is only performed on the complete regions

            sum_loss = gen_loss + gen_loss_mask + gen_disc_loss
            print(f'Train Epoch {epoch} Iter {it} Gen: {sum_loss.item():.4f}, {gen_loss.item():.4f}, {gen_loss_mask.item():.4f}, {gen_disc_loss.item():.4f}, 0')
            sum_loss.backward()
            optimizer.step()

            ######### Discriminator Phase #########
            disc_optimizer.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = disc_loss_fn(gt_disc_out, valid)
            fake_loss = disc_loss_fn(disc((location, pred.detach(), batch_size)), fake)
            disc_loss = (real_loss + fake_loss) / 2
            print(f'Train Epoch {epoch} Iter {it} Disc: {disc_loss.item():.4f}')
            disc_loss.backward()
            disc_optimizer.step()

            if it % 1000 == 0:
                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'{ckpt_savefile}/ckpt_{epoch}_{it}.pt')
            
            if it % 20 == 0:
                if val_data_loader is not None:
                    val_data = next(val_data_iter, None)
                    if val_data is None:
                        val_data_iter = iter(val_data_loader)
                        val_data = next(val_data_iter, None)
                    val_model(val_data, model, loss_fn, disc, disc_loss_fn)
            
            sys.stdout.flush()

    return

def val_model(data, model, loss_fn, disc, disc_loss_fn):
    
    with torch.no_grad():

        batch = data['batch'].long().cuda()
        vertex_idx = data['vertex_idx'].long().cuda()
        vertex_input = data['vertex_input'].float().cuda()
        vertex_output = data['vertex_output'].float().cuda()
        mask = (1 - vertex_input[..., -1]).bool()

        batch_size = batch.max().item() + 1
        location = torch.cat([vertex_idx, batch], dim=-1).long().contiguous()
        pred = model((location, vertex_input, batch_size))

        gen_loss = criterion(pred, vertex_output)

        gen_loss_mask = criterion(pred[mask], vertex_output[mask])

        gt_disc_out = disc((location, vertex_output, batch_size))
        valid = torch.ones_like(gt_disc_out, requires_grad=False).cuda()
        fake = torch.zeros_like(gt_disc_out, requires_grad=False).cuda()

        # Loss measures generator's ability to fool the discriminator
        gen_disc_out = disc((location, pred, batch_size))
        gen_disc_loss = disc_loss_fn(gen_disc_out, valid)

        sum_loss = gen_loss + gen_loss_mask + gen_disc_loss
        print(f'Val Gen: {sum_loss.item():.4f}, {gen_loss.item():.4f}, {gen_loss_mask.item():.4f}, {gen_disc_loss.item():.4f}')

if __name__ == '__main__':

    rm_sid = [13,14,19,20,21,42]
    all_sid = [i for i in range(48) if i not in rm_sid]
    val_sid = [0,1,34,35,36]
    train_sid = [i for i in all_sid if i not in val_sid]

    npz_folder = '../../replica_habitat_gen/ReplicaGenFtTriplets/easy/npz_regdim4emb32in'
    train_data = ReplicaGenDataset(npz_folder, train_sid)
    val_data = ReplicaGenDataset(npz_folder, val_sid)

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_data, batch_size=5, shuffle=True, collate_fn=collate_fn)

    device = 'cuda:0'
    scn_unet = SparseConvUNet(5, 4, torch.Tensor([144, 64, 160]).long(), reps=2, n_planes=[i * 32 for i in [1, 2, 3, 4, 5]]).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(scn_unet.parameters(), lr=1e-4)

    disc = Discriminator(4).to(device)
    disc_criterion = nn.MSELoss()
    disc_optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)

    train_model(train_dataloader, 
                scn_unet, criterion, optimizer, 
                disc, disc_criterion, disc_optimizer, 
                None, val_dataloader)

