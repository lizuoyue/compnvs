import torch
import numpy as np
import os, sys, tqdm, glob, random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import GeoGeneratorDF

class ReplicaRegenGeoDistanceFieldDataset(Dataset):

    def __init__(self, npz_folder, max_spatial):
        self.files = sorted(glob.glob(npz_folder + '/*.npz'))
        df_folder = npz_folder + '_df'
        self.df_files =  sorted(glob.glob(df_folder + '/*.npz'))
        self.max_spatial = np.array(max_spatial)
        return
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        df_files = self.df_files[idx * 3: idx * 3 + 3]

        npz = np.load(file, allow_pickle=True)
        geometry = torch.from_numpy(npz['geometry'])
        mask = torch.from_numpy(npz['mask']).bool()

        # geo  mask     ch1 ch2 y
        #   0	True	0.5 1	0
        #   1	True	0.5 1	1
        #   0	False   0	0	0
        #   1	False   1	0	1

        df_i = torch.from_numpy(np.load(df_files[0], allow_pickle=True)['distance_field'])
        df_j = torch.from_numpy(np.load(df_files[2], allow_pickle=True)['distance_field'])
        df_k = torch.from_numpy(np.load(df_files[1], allow_pickle=True)['distance_field'])

        df_out = torch.sqrt(torch.minimum(df_k, torch.minimum(df_i, df_j)).float()) # dm
        df_in = df_out + 0.0
        df_in[mask] = -1.0
        df_in = torch.stack([df_in, mask.float()])

        d = {}
        d['df_in'] = df_in
        d['df_out'] = df_out
        d['mask'] = mask
        d['num_0'] = (1 - geometry[mask]).sum()
        d['num_1'] = geometry[mask].sum()
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


def train_model(model, data_loader, loss_fn, optimizer, ckpt_path=None, val_data_loader=None):
    if ckpt_path is not None:
        d = torch.load(ckpt_path)
        model.load_state_dict(d['model_state_dict'])
        optimizer.load_state_dict(d['optimizer_state_dict'])
    
    if val_data_loader is not None:
        val_data_iter = iter(val_data_loader)

    assert(len(loss_fn) == 2)
    df_loss_fn, geo_loss_fn = loss_fn
    
    for epoch in range(100):
        for it, data in enumerate(tqdm.tqdm(data_loader)):

            df_in = data['df_in'].float().cuda()
            df_out = data['df_out'].float().cuda()
            mask = data['mask'].cuda()
            n0 = data['num_0'].sum()
            n1 = data['num_1'].sum()
            idx = data['idx']
            geo_y = (df_out < 0.5).float()

            optimizer.zero_grad()
            pred_geo_out, pred_df_out = model(df_in)
            pred_df_out = torch.abs(pred_df_out[:, 0])
            pred_df_out = (torch.tanh(pred_df_out) + 1.0) / 2.0 * 15.9687194227
            pred_geo_out = pred_geo_out[:, 0]

            df_loss = df_loss_fn(pred_df_out[mask], df_out[mask])
            df_weight = 1.0 / (df_out[mask] + 1.0)
            df_weight = df_weight / df_weight.sum()
            df_loss = torch.sum(df_weight * df_loss)

            geo_loss = geo_loss_fn(pred_geo_out[mask], geo_y[mask])
            geo_weight = geo_y[mask]/n1 * 1/2 + (1-geo_y[mask])/n0 * 1/2
            geo_loss = torch.sum(geo_weight * geo_loss)

            loss = df_loss + geo_loss

            pred_geo_y = torch.nn.Sigmoid()(pred_geo_out) > 0.5
            pd_bool = pred_geo_y[mask].float()
            gt_bool = geo_y[mask].bool()
            recall = pd_bool[gt_bool].mean()

            print(f'Epoch {epoch} Iter {it} {loss.item():6f} {df_loss.item():6f} {geo_loss.item():6f} {recall.item():3f}', idx.cpu().numpy())

            loss.backward()
            optimizer.step()

            if it % 1000 == 0:

                d = {
                    'epoch_it': (epoch, it),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(d, f'ckpt_replica_geo_df/ckpt_{epoch}_{it}.pt')
            
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

        df_in = data['df_in'].float().cuda()
        df_out = data['df_out'].float().cuda()
        mask = data['mask'].cuda()
        n0 = data['num_0'].sum()
        n1 = data['num_1'].sum()
        idx = data['idx']
        geo_y = (df_out < 0.5).float()

        pred_geo_out, pred_df_out = model(df_in)
        pred_df_out = torch.abs(pred_df_out[:, 0])
        pred_df_out = (torch.tanh(pred_df_out) + 1.0) / 2.0 * 15.9687194227
        pred_geo_out = pred_geo_out[:, 0]

        df_loss = df_loss_fn(pred_df_out[mask], df_out[mask])
        df_weight = 1.0 / (df_out[mask] + 1.0)
        df_weight = df_weight / df_weight.sum()
        df_loss = torch.sum(df_weight * df_loss)

        geo_loss = geo_loss_fn(pred_geo_out[mask], geo_y[mask])
        geo_weight = geo_y[mask]/n1 * 1/2 + (1-geo_y[mask])/n0 * 1/2
        geo_loss = torch.sum(geo_weight * geo_loss)

        loss = df_loss + geo_loss

        pred_geo_y = torch.nn.Sigmoid()(pred_geo_out) > 0.5
        pd_bool = pred_geo_y[mask].float()
        gt_bool = geo_y[mask].bool()
        recall = pd_bool[gt_bool].mean()

        iou, p, r = compute_iou_pr(pred_geo_y[mask], geo_y[mask])

        print(f'Val Loss {loss.item():6f} {df_loss.item():6f} {geo_loss.item():6f}')
        print(f'Val IoU {iou:6f}')
        print(f'Val Pre {p:6f}')
        print(f'Val Rec {r:6f}')



def test_model(model, data_loader, loss_fn, ckpt_path):

    assert(len(loss_fn) == 2)
    df_loss_fn, geo_loss_fn = loss_fn

    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    
    with torch.no_grad():

        for it, data in enumerate(tqdm.tqdm(data_loader)):

            df_in = data['df_in'].float().cuda()
            df_out = data['df_out'].float().cuda()
            mask = data['mask'].cuda()
            n0 = data['num_0'].sum()
            n1 = data['num_1'].sum()
            idx = data['idx']
            geo_y = (df_out < 0.5).float()

            pred_geo_out, pred_df_out = model(df_in)
            pred_df_out = torch.abs(pred_df_out[:, 0])
            pred_df_out = (torch.tanh(pred_df_out) + 1.0) / 2.0 * 15.9687194227
            pred_geo_out = pred_geo_out[:, 0]

            df_loss = df_loss_fn(pred_df_out[mask], df_out[mask])
            df_weight = 1.0 / (df_out[mask] + 1.0)
            df_weight = df_weight / df_weight.sum()
            df_loss = torch.sum(df_weight * df_loss)

            geo_loss = geo_loss_fn(pred_geo_out[mask], geo_y[mask])
            geo_weight = geo_y[mask]/n1 * 1/2 + (1-geo_y[mask])/n0 * 1/2
            geo_loss = torch.sum(geo_weight * geo_loss)

            loss = df_loss + geo_loss

            pred_geo_y = torch.nn.Sigmoid()(pred_geo_out) > 0.9
            pd_bool = pred_geo_y[mask].float()
            gt_bool = geo_y[mask].bool()
            recall = pd_bool[gt_bool].mean()

            iou, p, r = compute_iou_pr(pred_geo_y[mask], geo_y[mask])

            print(f'Val Loss {loss.item():6f} {df_loss.item():6f} {geo_loss.item():6f}')
            print(f'Val IoU {iou:6f}')
            print(f'Val Pre {p:6f}')
            print(f'Val Rec {r:6f}')

            if True:
                for i in range(idx.shape[0]):
                    with open(f'pts_dff/{idx[i]}_ch1_1.txt', 'w') as f:
                        tar = df_in[i, 0].cpu().numpy()
                        for pt in np.stack(np.nonzero((-0.5 < tar) & (tar < 0.5)), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

                    with open(f'pts_dff/{idx[i]}_mask.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple

                    with open(f'pts_dff/{idx[i]}_y_1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((geo_y[i].cpu().numpy() == 1) & mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

                    with open(f'pts_dff/{idx[i]}_pred_y_1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((pred_geo_y[i].cpu().numpy() == 1) & mask[i].cpu().numpy()), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    
                    with open(f'pts_dff/{idx[i]}_pred_y_1_all.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((pred_geo_y[i].cpu().numpy() == 1)), axis=-1):
                            f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange

            break




def check_triplet(idx):
    from PIL import Image
    filename = sorted(glob.glob('/home/lzq/lzy/replica_habitat_gen/ReplicaGeoTriplets/train_new/*.npz'))[idx]
    sid, fk, fi, fj = os.path.basename(filename.replace('.npz', '')).split('_')
    ik = np.array(Image.open(f'/home/lzq/lzy/replica_habitat_gen/ReplicaGen/scene{sid}/rgb/{fk}.png'))
    ii = np.array(Image.open(f'/home/lzq/lzy/replica_habitat_gen/ReplicaGen/scene{sid}/rgb/{fi}.png'))
    ij = np.array(Image.open(f'/home/lzq/lzy/replica_habitat_gen/ReplicaGen/scene{sid}/rgb/{fj}.png'))
    dk = np.load(f'/home/lzq/lzy/replica_habitat_gen/RawData/depth/train/{(int(sid)*300+int(fk)):05d}.npz')['depth']
    di = np.load(f'/home/lzq/lzy/replica_habitat_gen/RawData/depth/train/{(int(sid)*300+int(fi)):05d}.npz')['depth']
    dj = np.load(f'/home/lzq/lzy/replica_habitat_gen/RawData/depth/train/{(int(sid)*300+int(fj)):05d}.npz')['depth']
    import matplotlib; matplotlib.use('agg')
    import matplotlib.pyplot as plt
    to_show_1 = np.hstack([ik, ii, ij])
    to_show_2 = np.hstack([dk, di, dj])
    plt.imshow(to_show_1)
    plt.savefig('triple_img.png')
    plt.imshow(to_show_2)
    plt.savefig('triple_dep.png')
    quit()

if __name__ == '__main__':

    train_data = ReplicaRegenGeoDistanceFieldDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaRegenGeo/train', [144,64,160])
    val_data = ReplicaRegenGeoDistanceFieldDataset('/home/lzq/lzy/replica_habitat_gen/ReplicaRegenGeo/val', [144,64,160])

    print('Train:', len(train_data))
    print('Val:', len(val_data))

    train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=4, shuffle=False)

    geo_generator = GeoGeneratorDF(nf_in_geo=2, nf=64, max_data_size=[144,64,160]).cuda()
    loss_fn = (torch.nn.SmoothL1Loss(), torch.nn.BCEWithLogitsLoss(reduction='none'))
    optimizer = torch.optim.Adam(geo_generator.parameters(), lr=1e-4)

    # train_model(geo_generator, train_dataloader, loss_fn, optimizer, None, val_dataloader)

    test_model(geo_generator, val_dataloader, loss_fn, 'ckpt_replica_geo_df/ckpt_2_3000.pt')
