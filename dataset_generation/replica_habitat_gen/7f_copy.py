import numpy as np
import tqdm, random
from PIL import Image
import torch
import os, glob, sys
import MinkowskiEngine as ME
sys.path.append('/home/lzq/lzy/minkowski_3d_completion')
from minkowski_utils import replace_features
from utils import project_depth
import scipy.ndimage as mor

if __name__ == '__main__':

    stage = 'mid'
    n_scene = 48
    voxel_size = 0.1
    a, b = int(sys.argv[1]), int(sys.argv[2])

    for sid in range(a, b):

        triplets = np.load(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz')[stage]
        
        for k, i, j in tqdm.tqdm(triplets):

            basename = f'{sid:02d}_{k:03d}_{i:03d}_{j:03d}'

            src = f'ReplicaGen/scene{sid:02d}/rgb/{k:03d}.png'
            tar = f'ReplicaGenEncFtTriplets/mid/rgb/{basename}.png'
            os.system(f'cp {src} {tar}')

            src = f'ReplicaGen/scene{sid:02d}/depth/{k:03d}.npz'
            tar = f'ReplicaGenEncFtTriplets/mid/depth/{basename}.npz'
            os.system(f'cp {src} {tar}')

            src = f'ReplicaGen/scene{sid:02d}/pose/{k:03d}.txt'
            tar = f'ReplicaGenEncFtTriplets/mid/pose/{basename}.txt'
            os.system(f'cp {src} {tar}')
            # dict(np.load(f'ReplicaGenEncFtTriplets/{stage}/npz_basic_prednoout/{basename}.npz', 
            

            # src = np.load(f'ReplicaGenEncFtTriplets/{stage}/npz_norgba_pred/{basename}.npz', allow_pickle=True)
            # tar = dict(np.load(f'ReplicaGenEncFtTriplets/{stage}/npz_basic_prednoout/{basename}.npz', allow_pickle=True))
            # assert(src['vertex_input'].shape[0] == src['vertex_output'].shape[0])
            # assert(src['vertex_input'].shape[1] == 33)
            # assert(src['vertex_output'].shape[1] == 33)
            # print(src[''])
            # quit()
            
            # tar['vertex_output'] = src['vertex_output']
            # tar['reference'] = src['reference']
            # np.savez_compressed(f'ReplicaGenEncFtTriplets/{stage}/npz_basic_pred/{basename}.npz', **tar)
            # assert((src['vertex_idx'] == tar['vertex_idx']).mean() == 1.0)








