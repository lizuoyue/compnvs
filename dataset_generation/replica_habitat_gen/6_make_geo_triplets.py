import numpy as np
import tqdm, random, sys
from PIL import Image
from utils import project, OccupancyGrid, PseudoColorConverter
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

# 0: free space
# 1: occupied
# 2: in-view unobserved
# 3: out-view unobserved

# combine i & j
# i\j 0 1 2 3
# 0   0 1 0 0
# 1   1 1 1 1
# 2   0 1 2 2
# 3   0 1 2 3

#   ch0 ch1 ch2
# 0  1   0   0
# 1  0   1   0
# 2  0   0   1
# 3  0   0   0

# total: 43149
# train: 37749
# val: 5400

if __name__ == '__main__':
    
    stage = 'easy'
    spatial_size = [144, 64, 160] # center
    voxel_size = 0.1

    geo_comb = np.array([
        [0, 1, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 2, 2],
        [0, 1, 2, 3],
    ], np.int32)

    geo_ch = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ], np.float32)

    a, b = int(sys.argv[1]), int(sys.argv[2])

    val_sid = [13,14,19,20,21,42]
    val_sid = [sid for sid in range(a, b) if sid in val_sid]
    train_sid = [sid for sid in range(a, b) if sid not in val_sid]

    cmap = PseudoColorConverter('viridis', 0.0, 0.5)

    for phase, sid_li in zip([f'train_{stage}', f'val_{stage}'], [train_sid, val_sid]):
        for sid in sid_li:

            for i in tqdm.tqdm(list(range(300))):
                scene_grid = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + i):05d}.npz')['grid']
                scene_df = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300 + i):05d}.npz')['distance_field']
                print(scene_df[scene_grid == 1].min(), scene_df[scene_grid == 1].max(), (scene_df[scene_grid == 1]<=0.07071067811).mean())
            quit()

            scene_grid = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300):05d}.npz')['grid']
            for i in tqdm.tqdm(list(range(1, 300))):
                addition = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + i):05d}.npz')['grid']
                scene_grid = geo_comb[scene_grid.flatten(), addition.flatten()].reshape(scene_grid.shape)
            
            scene_df = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300):05d}.npz')['distance_field']
            for i in tqdm.tqdm(list(range(1, 300))):
                addition = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300 + i):05d}.npz')['distance_field']
                scene_df = np.minimum(scene_df, addition)
            # scene_df = np.sqrt(scene_df.astype(np.float32))

            if False:
                pts = np.stack(np.nonzero(scene_df <= 0.5), axis=-1)
                rgb = cmap.convert(scene_df[scene_df <= 0.5])
                with open('vis_df.txt', 'w') as f:
                    for pt, (r, g, b) in zip(pts, rgb):
                        f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
                
            if False:
                with open(f'pts/scene_occ_pts_{sid:02d}_in_fs.txt', 'w') as f:
                    for pt in np.stack(np.nonzero(scene_grid == 0), axis=-1):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                with open(f'pts/scene_occ_pts_{sid:02d}_in_occ.txt', 'w') as f:
                    for pt in np.stack(np.nonzero(scene_grid == 1), axis=-1):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                with open(f'pts/scene_occ_pts_{sid:02d}_in_uob.txt', 'w') as f:
                    for pt in np.stack(np.nonzero(scene_grid == 2), axis=-1):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                with open(f'pts/scene_occ_pts_{sid:02d}_out_uob.txt', 'w') as f:
                    for pt in np.stack(np.nonzero(scene_grid == 3), axis=-1):
                        f.write('%.2lf;%.2lf;%.2lf;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
            
            triplets = np.load(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz')[stage]
            for k, i, j in tqdm.tqdm(triplets):
                oc_i = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + i):05d}.npz')['grid']
                oc_j = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + j):05d}.npz')['grid']
                # oc_ij = geo_comb[oc_i.flatten(), oc_j.flatten()].reshape(spatial_size)
                # ch012 = geo_ch[oc_ij.flatten()].reshape(spatial_size + [3])
                oc_input = geo_comb[oc_i.flatten(), oc_j.flatten()].reshape(spatial_size)

                df_i = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300 + i):05d}.npz')['distance_field']
                df_j = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300 + j):05d}.npz')['distance_field']
                # ch3 = np.sqrt(np.minimum(df_i, df_j).astype(np.float32)) / 10.0 # from 0.0 to 1.596872
                df_input = np.sqrt(np.minimum(df_i, df_j).astype(np.float32))

                oc_k = np.load(f'ReplicaGenGeometry/occupancy/{(sid * 300 + k):05d}.npz')['grid']
                df_k = np.load(f'ReplicaGenGeometry/distance_field/{(sid * 300 + k):05d}.npz')['distance_field']
                
                mask_roi = (oc_i <= 2) | (oc_j <= 2) | (oc_k <= 2)
                oc_output = scene_grid.copy()
                oc_output[~mask_roi] = 3

                df_output = scene_df.copy()
                df_output[~mask_roi] = -1.0

                mask_gen = (oc_k <= 2) & (oc_input >= 2)

                # mask = (oc_ij >= 2) & (oc_k <= 2)
                # ch4 = mask.astype(np.float32)

                # input_ch = np.concatenate([ch012, ch3[..., np.newaxis], ch4[..., np.newaxis]], axis=-1)
                # output_label = geo_comb[oc_ij.flatten(), oc_k.flatten()].reshape(spatial_size)
                # output_df = np.minimum(ch3, np.sqrt(df_k.astype(np.float32)) / 10.0)

                # check = np.unique(output_label[ch4.astype(np.bool)])
                # assert(check.shape[0] <= 3)
                # assert(check.min() >= 0)
                # assert(check.max() <= 2)

                if False:
                    with open(f'pts/pts_ch0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(input_ch[..., 0]), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts/pts_ch1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(input_ch[..., 1]), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    with open(f'pts/pts_ch2.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(input_ch[..., 2]), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                    with open(f'pts/pts_ch4.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(input_ch[..., 4]), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
                    with open(f'pts/pts_out0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((output_label == 0) & mask), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts/pts_out1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((output_label == 1) & mask), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    with open(f'pts/pts_out2.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((output_label == 2) & mask), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                    quit()
                
                if False:
                    with open(f'pts/pts_ch0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(oc_input==0), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts/pts_ch1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(oc_input==1), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    with open(f'pts/pts_ch2.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(oc_input==2), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                    with open(f'pts/pts_gen.txt', 'w') as f:
                        for pt in np.stack(np.nonzero(mask_gen), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [148,103,189])) # Tableau purple
                    with open(f'pts/pts_out0.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((oc_output == 0) & mask_gen), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [31,119,180])) # Tableau blue
                    with open(f'pts/pts_out1.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((oc_output == 1) & mask_gen), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [255,127,14])) # Tableau orange
                    with open(f'pts/pts_out2.txt', 'w') as f:
                        for pt in np.stack(np.nonzero((oc_output == 2) & mask_gen), axis=-1):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [127,127,127])) # Tableau gray
                    
                    pts = np.stack(np.nonzero(df_input <= 0.5), axis=-1)
                    rgb = cmap.convert(df_input[df_input <= 0.5])
                    with open('pts/pts_df_in.txt', 'w') as f:
                        for pt, (r, g, b) in zip(pts, rgb):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
                    
                    vis_mask = (0.0 <= df_output) & (df_output <= 0.5)
                    pts = np.stack(np.nonzero(vis_mask), axis=-1)
                    rgb = cmap.convert(df_output[vis_mask])
                    with open('pts/pts_df_out.txt', 'w') as f:
                        for pt, (r, g, b) in zip(pts, rgb):
                            f.write('%d;%d;%d;%d;%d;%d\n' % tuple(list(pt) + [int(r),int(g),int(b)]))
                    quit()

                # np.savez_compressed(f'ReplicaGenGeoTriplets/{phase}/{sid:02d}_{k:03d}_{i:03d}_{j:03d}.npz',
                #     input_ch=input_ch,
                #     mask=mask,
                #     output_lb=output_label,
                #     output_df=output_df,
                # )

                np.savez_compressed(f'ReplicaGenGeoTriplets/{phase}/{sid:02d}_{k:03d}_{i:03d}_{j:03d}.npz',
                    input_oc=oc_input,
                    input_df=df_input,
                    output_oc=oc_output,
                    output_df=df_output,
                    mask_roi=mask_roi,
                    mask_gen=mask_gen,
                )
