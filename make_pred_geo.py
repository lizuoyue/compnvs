import numpy as np
import tqdm, random
from PIL import Image
import torch
import os, glob, sys
import MinkowskiEngine as ME
from geo_completion.minkowski_utils import replace_features
# import scipy.ndimage.morphology as mor

def load(npz_file):
    npz = np.load(npz_file, allow_pickle=True)
    idx = torch.from_numpy(npz['vertex'])
    idx = torch.cat([idx[:,:1] * 0, idx], dim=-1)
    valid = (np.abs(npz['values']).sum(axis=-1) > 1e-6).astype(np.float32)
    val = np.concatenate([npz['values'], valid[:,np.newaxis]], axis=-1)
    val = torch.from_numpy(val)
    return idx, val

offset_np = np.array([
    [-1,-1,-1],
    [-1,-1, 1],
    [-1, 1,-1],
    [-1, 1, 1],
    [ 1,-1,-1],
    [ 1,-1, 1],
    [ 1, 1,-1],
    [ 1, 1, 1],
], np.int)

def load_pred(file):
    pred_geo = np.load(file)['pred_geo']

    # kernel = np.array([[1,1,1],[1,1,1],[1,1,1]])
    # kernel_x = kernel[np.newaxis]
    # kernel_y = kernel[:,np.newaxis]
    # kernel_z = kernel[:,:,np.newaxis]

    # pred_min, pred_max = pred_geo.min(axis=0) - 5, pred_geo.max(axis=0) + 5
    # pred_biased = pred_geo - pred_min
    # dense_grid = np.zeros(pred_max - pred_min + 1, np.bool)
    # dense_grid[pred_biased[:,0], pred_biased[:,1], pred_biased[:,2]] = True
    # dense_grid_x = mor.binary_closing(dense_grid, structure=kernel_x, iterations=3)
    # dense_grid_y = mor.binary_closing(dense_grid, structure=kernel_y, iterations=3)
    # dense_grid_z = mor.binary_closing(dense_grid, structure=kernel_z, iterations=3)
    # dense_grid = dense_grid_x | dense_grid_y | dense_grid_z

    # np.savetxt(f'closing_test/{idx}_pre.txt', pred_geo, '%d')
    # np.savetxt(f'closing_test/{idx}_postx.txt', np.stack(dense_grid_x.nonzero(), axis=1)+pred_min, '%d')
    # np.savetxt(f'closing_test/{idx}_posty.txt', np.stack(dense_grid_y.nonzero(), axis=1)+pred_min, '%d')
    # np.savetxt(f'closing_test/{idx}_postz.txt', np.stack(dense_grid_z.nonzero(), axis=1)+pred_min, '%d')
    # input()
    # pred_geo = np.stack(dense_grid.nonzero(), axis=1) + pred_min

    vertex_idx = np.repeat(pred_geo * 2 + 1, 8, axis=0) + np.tile(offset_np, (pred_geo.shape[0], 1)) # all voxel vertices index, always even
    vertex_idx = (np.unique(vertex_idx, axis=0) / 2).astype(np.int32)
    idx = torch.from_numpy(vertex_idx)
    idx = torch.cat([idx[:,:1] * 0, idx], dim=-1)
    return idx



if __name__ == '__main__':


    pool = ME.MinkowskiSumPooling(kernel_size=2, stride=1, dimension=3)
    offset = torch.from_numpy(np.array([
        [-1,-1,-1],
        [-1,-1, 1],
        [-1, 1,-1],
        [-1, 1, 1],
        [ 1,-1,-1],
        [ 1,-1, 1],
        [ 1, 1,-1],
        [ 1, 1, 1],
    ], np.int))

    files = sorted(glob.glob('ExampleScenesFused/*/npz/00000.npz'))
    
    with torch.no_grad():
        for it, file in tqdm.tqdm(list(enumerate(files))):

            idx_ij, val_ij = load(file.replace('/npz/', '/npz_fts/'))

            idx_k = load_pred(f'geo_completion/mink_comp_pred/{it}_pred.npz')
            val_k = torch.zeros(idx_k.shape[0], val_ij.shape[1]).float()

            idx_ijk = torch.cat([idx_ij, idx_k], dim=0)
            val_ij0 = torch.cat([val_ij, val_k], dim=0)

            input_info = ME.SparseTensor(
                coordinates=idx_ijk,
                features=val_ij0,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_SUM,
            )

            # output_info = ME.SparseTensor(
            #     coordinates=idx_tri,
            #     features=val_tri,
            #     coordinate_manager=input_info.coordinate_manager,
            # )

            mink_ones = replace_features(input_info, input_info.F[:,:1] * 0 + 1)
            center_nums = pool(mink_ones)
            center_idx = center_nums.C[torch.round(center_nums.F[:,0]) == 8, 1:].float() + 0.5 # half stride

            # TODO: Here batch size do not support
            vertex_idx = torch.repeat_interleave(center_idx, 8, dim=0) + 0.5 * offset.repeat(center_idx.shape[0], 1)
            vertex_idx = torch.round(vertex_idx).int()
            vertex_idx, center_to_vertex = torch.unique(vertex_idx, dim=0, return_inverse=True)
            
            center_to_vertex = center_to_vertex.reshape(-1, 8)
            center_pts = center_idx * 0.1

            locations = torch.cat([vertex_idx[:,:1]*0, vertex_idx], dim=1).float()
            vertex_input = input_info.features_at_coordinates(locations)
            vertex_input = vertex_input / torch.maximum(vertex_input[:,-1:], vertex_input[:,-1:]*0+1)
            vertex_output = vertex_input#output_info.features_at_coordinates(locations)

            # src_rgb = f'ARKitScenes/ARKitScenes/ARKitScenesSingleViewNSVF/all/rgb/{scan_id}_{tk}.png'
            # tar_rgb = f'ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/rgb/{basename}.png'
            # src_dep = f'ARKitScenes/ARKitScenes/ARKitScenesSingleViewNSVF/all/depth/{scan_id}_{tk}.npz'
            # tar_dep = f'ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/depth/{basename}.npz'
            # src_pose = f'ARKitScenes/ARKitScenes/ARKitScenesSingleViewNSVF/all/pose/{scan_id}_{tk}.txt'
            # tar_pose = f'ARKitScenes/ARKitScenes/ARKitScenesTripletsNSVF/all/pose/{basename}.txt'
            # os.system(f'cp {src_rgb} {tar_rgb}')
            # os.system(f'cp {src_dep} {tar_dep}')
            # os.system(f'cp {src_pose} {tar_pose}')

            #
            np.savez_compressed(
                file.replace('/npz/', '/npz_fts_predgeo/'),
                center_pts=center_pts.numpy().astype(np.float32),
                center_to_vertex=center_to_vertex.numpy().astype(np.int32),
                vertex_idx=vertex_idx.numpy().astype(np.int32),
                vertex_input=vertex_input.numpy().astype(np.float32), # last channel mask
                vertex_output=vertex_output.numpy().astype(np.float32),
                reference=None,
            )

