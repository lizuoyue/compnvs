import torch
import numpy as np
import os, sys, tqdm, glob, random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GeoGenerator

class ReplicaGenGeoDataset(Dataset):

    def __init__(self, npz_folder, max_spatial):
        self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        self.max_spatial = np.array(max_spatial)
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        grid = torch.from_numpy(np.load(file, allow_pickle=True)['geometry'])

        # grid  ch1 ch2    mask  y
        #  -1	0.5	FALSE	0	99
        #   0	0	FALSE	0	0
        #   1	1	FALSE	0	1
        #  99	0.5	TRUE	0	99
        # 100	0.5	TRUE	1	0
        # 101	0.5	TRUE	1	1

        ch2 = (grid % 200 > 10).float()
        ch1 = ((1 - ch2) * grid).float() + ch2 * 0.5
        ch2[grid < 0] *= 0
        d = {}
        d['geo_x'] = torch.stack([ch1, ch2])
        d['geo_y'] = (grid % 100).int() # 99 unobserved
        d['mask'] = (grid >= 100).bool()
        d['mask_view'] = (grid >= 99).bool()
        d['num_0'] = (1 - d['geo_y'][d['mask']]).sum()
        d['num_1'] = d['geo_y'][d['mask']].sum()
        d['idx'] = idx

        if False:
            with open(f'pts_multi/{idx}_ch1_0.txt', 'w') as f:
                for pt in np.stack(np.nonzero(d['geo_x'][0].numpy() < 0.1), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts_multi/{idx}_ch1_1.txt', 'w') as f:
                for pt in np.stack(np.nonzero(d['geo_x'][0].numpy() > 0.9), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

            with open(f'pts_multi/{idx}_mask.txt', 'w') as f:
                for pt in np.stack(np.nonzero(d['mask'].numpy()), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple

            with open(f'pts_multi/{idx}_y_0.txt', 'w') as f:
                for pt in np.stack(np.nonzero((d['geo_y'].numpy() == 0) & d['mask'].numpy()), axis=-1):
                    f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
            with open(f'pts_multi/{idx}_y_1.txt', 'w') as f:
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


def train_model(model, data_loader, loss_fn, optimizer, ckpt_path=None, val_data_loader=None):
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)
    
    for epoch in range(100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            geo_x = data['geo_x'].float().cuda()
            geo_y = data['geo_y'].float().cuda()
            mask = data['mask'].cuda()
            n0 = data['num_0'].sum()
            n1 = data['num_1'].sum()
            idx = data['idx']

            if n0 == 0 or n1 == 0:
                continue

            optimizer.zero_grad()
            pred_geo_y = model(geo_x)[:, 0]
            loss = loss_fn(pred_geo_y[mask], geo_y[mask])
            weight = geo_y[mask]/n1 * 1/2 + (1-geo_y[mask])/n0 * 1/2
            loss = torch.sum(weight * loss)

            pred_geo_y_01 = torch.nn.Sigmoid()(pred_geo_y) > 0.5
            pd_bool = pred_geo_y_01[mask].float()
            gt_bool = geo_y[mask].bool()
            recall = pd_bool[gt_bool].mean()

            print(f'Epoch {epoch} Iter {it}', loss.item(), recall.item(), idx.cpu().numpy())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'ckpt_replica_geo_multi/ckpt_{epoch}_{it}.pt')
            
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

        geo_x = data['geo_x'].float().cuda()
        geo_y = data['geo_y'].float().cuda()
        mask = data['mask'].cuda()
        n0 = data['num_0'].sum()
        n1 = data['num_1'].sum()
        idx = data['idx']

        if n0 == 0 or n1 == 0:
            return

        pred_geo_y = model(geo_x)[:, 0]
        loss = loss_fn(pred_geo_y[mask], geo_y[mask])
        pred_prob_y = torch.nn.Sigmoid()(pred_geo_y) > 0.5
        iou, p, r = compute_iou_pr(pred_prob_y[mask], geo_y[mask])

        weight = geo_y[mask]/n1/2 + (1-geo_y[mask])/n0/2
        loss = torch.sum(weight * loss)

        print(f'Val Loss {loss.item():6f}')
        print(f'Val IoU {iou:6f}')
        print(f'Val Pre {p:6f}')
        print(f'Val Rec {r:6f}')



def test_model(model, data_loader, loss_fn, ckpt_path):

    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            geo_x = data['geo_x'].float().cuda()
            geo_y = data['geo_y'].float().cuda()
            mask = data['mask'].cuda()
            mask_view = data['mask_view'].cuda()
            n0 = data['num_0'].sum()
            n1 = data['num_1'].sum()
            idx = data['idx']

            if n0 == 0 or n1 == 0:
                continue

            pred_geo_y = model(geo_x)[:, 0]
            loss = loss_fn(pred_geo_y[mask], geo_y[mask])
            pred_prob_y = torch.nn.Sigmoid()(pred_geo_y) > 0.5
            iou, p, r = compute_iou_pr(pred_prob_y[mask_view] == 1, geo_y[mask_view] == 1)

            weight = geo_y[mask]/n1/2 + (1-geo_y[mask])/n0/2
            loss = torch.sum(weight * loss)

            print(f'Loss {loss.item():6f}')
            print(f'IoU {iou:6f}')
            print(f'Pre {p:6f}')
            print(f'Rec {r:6f}')

            if True:
                for i in range(idx.shape[0]):
                    with open(f'pts_multi/{idx[i]}_ch1_0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(geo_x[i, 0].cpu().numpy() < 0.1), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts_multi/{idx[i]}_ch1_1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(geo_x[i, 0].cpu().numpy() > 0.9), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

                    with open(f'pts_multi/{idx[i]}_mask.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple

                    with open(f'pts_multi/{idx[i]}_y_0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((geo_y[i].cpu().numpy() == 0) & mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts_multi/{idx[i]}_y_1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((geo_y[i].cpu().numpy() == 1) & mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    
                    with open(f'pts_multi/{idx[i]}_pred_y_0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((pred_prob_y[i].cpu().numpy() == 0) & mask_view[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts_multi/{idx[i]}_pred_y_1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((pred_prob_y[i].cpu().numpy() == 1) & mask_view[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

            break













if __name__ == '__main__':

    train_data = ReplicaGenGeoDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGeoTriplets/train_new', [160,80,192])
    val_data = ReplicaGenGeoDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaGeoTriplets/val_new', [160,80,192])

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)

    geo_generator = GeoGenerator(nf_in_geo=2, nf=64, max_data_size=[160,80,192]).cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(geo_generator.parameters(), lr=1e-5)

    # train_model(geo_generator, train_dataloader, loss_fn, optimizer, 'ckpt_replica_geo_multi/save0/ckpt_14000.pt', val_dataloader)

    test_model(geo_generator, train_dataloader, loss_fn, 'ckpt_replica_geo_multi/ckpt_1_0.pt')
