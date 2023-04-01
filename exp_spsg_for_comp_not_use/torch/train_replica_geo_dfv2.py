import torch
import numpy as np
import os, sys, tqdm, glob, random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GeoGeneratorDFv2

class ReplicaRegenGeoDistanceFieldDataset(Dataset):

    def __init__(self, npz_folder, max_spatial):
        self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        self.max_spatial = np.array(max_spatial)
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz = np.load(self.files[idx], allow_pickle=True)
        input_ch = torch.from_numpy(npz['input_ch'].transpose([3,0,1,2])).float()
        mask = torch.from_numpy(npz['mask']).bool()
        output_lb = torch.from_numpy(npz['output_lb']).long()
        output_df = torch.from_numpy(npz['output_df']).float()

        d = {}
        d['input_ch'] = input_ch
        d['output_lb'] = output_lb
        d['output_df'] = output_df
        d['mask'] = mask
        d['num_0'] = (output_lb[mask] == 0).sum()
        d['num_1'] = (output_lb[mask] == 1).sum()
        d['num_2'] = (output_lb[mask] == 2).sum()
        d['idx'] = idx

        if False:
            with open(f'pts/{idx}_ch1_1.txt', 'w') as f:
                for pt in np.stack(np.nonzero(d['geo_x'][0].numpy() > 0.9), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

            with open(f'pts/{idx}_mask.txt', 'w') as f:
                for pt in np.stack(np.nonzero(d['mask'].numpy()), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple

            with open(f'pts/{idx}_y_0.txt', 'w') as f:
                for pt in np.stack(np.nonzero((d['geo_y'].numpy() == 0) & d['mask'].numpy()), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts/{idx}_y_1.txt', 'w') as f:
                for pt in np.stack(np.nonzero((d['geo_y'].numpy() == 1) & d['mask'].numpy()), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

        return d


def compute_iou_pr(pd, gt):
    assert(pd.shape == gt.shape)
    pd_bool = pd.cpu().numpy().astype(np.bool)
    gt_bool = gt.cpu().numpy().astype(np.bool)
    i = (pd_bool & gt_bool).sum()
    u = (pd_bool | gt_bool).sum()
    r = pd_bool[gt_bool].mean()
    p = gt_bool[pd_bool].mean()
    return i / u, p, r

def forward(model, data, df_loss_fn, geo_loss_fn):
    input_ch = data['input_ch'].cuda()
    output_lb = data['output_lb'].cuda()
    output_df = data['output_df'].cuda()
    mask = data['mask'].cuda()
    n0, n1, n2 = data['num_0'].sum().item(), data['num_1'].sum().item(), data['num_2'].sum().item()
    idx = data['idx']

    pred_geo_out, pred_df_out = model(input_ch)
    pred_df_out = (torch.tanh(pred_df_out[:, 0]) + 1.0) / 2.0 * 15.9687194227
    pred_geo_out = torch.nn.LogSoftmax(dim=1)(pred_geo_out).permute(0,2,3,4,1)

    df_loss = df_loss_fn(pred_df_out[mask], output_df[mask])
    df_weight = 1.0 / (output_df[mask] + 1.0)
    df_weight = df_weight / df_weight.sum()
    df_loss = torch.sum(df_weight * df_loss)

    geo_loss = geo_loss_fn(pred_geo_out[mask], output_lb[mask])
    geo_weight  = (output_lb[mask]==0).float()/max(n0,1e-3) * 1/3
    geo_weight += (output_lb[mask]==1).float()/max(n1,1e-3) * 1/3
    geo_weight += (output_lb[mask]==2).float()/max(n2,1e-3) * 1/3
    geo_loss = torch.sum(geo_weight * geo_loss)

    loss = df_loss + geo_loss

    pred_geo_y = torch.argmax(pred_geo_out, dim=-1)
    # pd_bool = (pred_geo_y[mask] == 1).float()
    # gt_bool = (output_lb[mask] == 1).bool()
    # recall = pd_bool[gt_bool].mean()
    iou, p, r = compute_iou_pr(pred_geo_y[mask] == 1, output_lb[mask] == 1)

    if True:
        for i in range(idx.shape[0]):
            with open(f'pts_dfv2/{idx[i]}_pts_ch0.txt', 'w') as f:
                for pt in np.stack(np.nonzero(input_ch[i, 0].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts_dfv2/{idx[i]}_pts_ch1.txt', 'w') as f:
                for pt in np.stack(np.nonzero(input_ch[i, 1].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
            with open(f'pts_dfv2/{idx[i]}_pts_ch2.txt', 'w') as f:
                for pt in np.stack(np.nonzero(input_ch[i, 2].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
            with open(f'pts_dfv2/{idx[i]}_pts_ch4.txt', 'w') as f:
                for pt in np.stack(np.nonzero(input_ch[i, 4].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
            with open(f'pts_dfv2/{idx[i]}_pts_out0.txt', 'w') as f:
                for pt in np.stack(np.nonzero((output_lb[i].cpu().numpy() == 0) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts_dfv2/{idx[i]}_pts_out1.txt', 'w') as f:
                for pt in np.stack(np.nonzero((output_lb[i].cpu().numpy() == 1) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
            with open(f'pts_dfv2/{idx[i]}_pts_out2.txt', 'w') as f:
                for pt in np.stack(np.nonzero((output_lb[i].cpu().numpy() == 2) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
            with open(f'pts_dfv2/{idx[i]}_pts_pred_out0.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pred_geo_y[i].cpu().numpy() == 0) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts_dfv2/{idx[i]}_pts_pred_out1.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pred_geo_y[i].cpu().numpy() == 1) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
            with open(f'pts_dfv2/{idx[i]}_pts_pred_out2.txt', 'w') as f:
                for pt in np.stack(np.nonzero((pred_geo_y[i].cpu().numpy() == 2) & mask[i].cpu().numpy()), axis=-1):
                    f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
            quit()

    return loss, df_loss, geo_loss, (iou, p, r), pred_geo_y, pred_df_out



def train_model(model, data_loader, loss_fn, optimizer, ckpt_path=None, val_data_loader=None):

    epoch_load, it_load = 0, -1
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
        epoch_load, it_load = d['epoch_it']
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)

    assert(len(loss_fn) == 2)
    df_loss_fn, geo_loss_fn = loss_fn
    
    for epoch in range(epoch_load, 100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            if epoch == epoch_load and it <= it_load:
                continue

            optimizer.zero_grad()
            loss, df_loss, geo_loss, (iou, p, r), pred_geo_y, pred_df_out = forward(model, data, df_loss_fn, geo_loss_fn)

            print(f'Epoch {epoch} Iter {it} {loss.item():.6f} {df_loss.item():.6f} {geo_loss.item():.6f} {r:.3f}', data['idx'].numpy())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'ckpt_replica_geo_dfv2/ckpt_{epoch}_{it}.pt')
            
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
    assert(len(loss_fn) == 2)
    df_loss_fn, geo_loss_fn = loss_fn
    with torch.no_grad():
        loss, df_loss, geo_loss, (iou, p, r), pred_geo_y, pred_df_out = forward(model, data, df_loss_fn, geo_loss_fn)
        print(f'Val Loss {loss.item():.6f} {df_loss.item():.6f} {geo_loss.item():.6f}')
        print(f'Val IoU {iou:.6f}')
        print(f'Val Pre {p:.6f}')
        print(f'Val Rec {r:.6f}')

def test_model(model, data_loader, loss_fn, ckpt_path):
    assert(len(loss_fn) == 2)
    df_loss_fn, geo_loss_fn = loss_fn

    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():

        for it, data in enumerate(tqdm.tqdm(data_loader)):
            loss, df_loss, geo_loss, (iou, p, r), pred_geo_y, pred_df_out = forward(model, data, df_loss_fn, geo_loss_fn)
            print(f'Test Loss {loss.item():.6f} {df_loss.item():.6f} {geo_loss.item():.6f}')
            print(f'Test IoU {iou:.6f}')
            print(f'Test Pre {p:.6f}')
            print(f'Test Rec {r:.6f}')
            break


if __name__ == '__main__':

    train_data = ReplicaRegenGeoDistanceFieldDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/train_easy', [144,64,160])
    val_data = ReplicaRegenGeoDistanceFieldDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGenGeoTriplets/val_easy', [144,64,160])

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)#, num_workers=24)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)#, num_workers=24)

    geo_generator = GeoGeneratorDFv2(nf_in_geo=5, nf=64, max_data_size=[144,64,160]).cuda()
    loss_fn = (torch.nn.SmoothL1Loss(), torch.nn.NLLLoss(reduction='none'))
    optimizer = torch.optim.Adam(geo_generator.parameters(), lr=1e-4)

    # train_model(geo_generator, train_dataloader, loss_fn, optimizer, None, val_dataloader)

    test_model(geo_generator, val_dataloader, loss_fn, 'ckpt_replica_geo_dfv2/ckpt_2_5000.pt')
