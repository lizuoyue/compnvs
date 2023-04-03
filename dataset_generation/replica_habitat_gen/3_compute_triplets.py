import numpy as np
import os, glob, tqdm
from PIL import Image

# Validation
# frl_apartment_0: 19 20 21
# apartment_1: 13 14
# office_2: 42

if __name__ == '__main__':

    np.random.seed(1993)

    n_scene = 48
    for sid in tqdm.tqdm(list(range(n_scene))):

        npz = np.load(f'ReplicaGenRelation/scene{sid:02d}.npz')
        mapping = npz['mapping']
        mat = npz['relation']
        frame_voxel_idx = npz['frame_voxel_idx']

        exclusive = []
        for i in range(299):
            for j in range(i + 1, 300):
                if mat[i, j] == 0 and mat[j, i] == 0:
                    exclusive.append((i, j))
        ij = np.array(exclusive)

        # nums = []
        easy_li, mid_li, hard_li = [], [], []
        for k in range(300):
            score_i = mat[ij[:,0], k]
            score_j = mat[ij[:,1], k]
            select = (score_i >= 0.05) & (score_j >= 0.05)

            ij_select = ij[select]
            score_i = score_i[select]
            score_j = score_j[select]
            score = score_i + score_j

            easy = (score_i <= 0.50) & (score_j <= 0.50) & (0.65 <= score) & (score < 0.70)
            mid  = (score_i <= 0.35) & (score_j <= 0.35) & (0.45 <= score) & (score < 0.50)
            hard = (score_i <= 0.20) & (score_j <= 0.20) & (0.25 <= score) & (score < 0.30)
            # print(k, easy.sum(), mid.sum(), hard.sum())
            # input()
            # nums.append(np.array([easy.sum(), mid.sum(), hard.sum()]))
            # continue

            easy = ij_select[np.nonzero(easy)[0]]
            mid  = ij_select[np.nonzero(mid )[0]]
            hard = ij_select[np.nonzero(hard)[0]]

            num_to_sel = 3
            for sel, li in zip([easy, mid, hard], [easy_li, mid_li, hard_li]):
                if sel.shape[0] >= num_to_sel:
                    for i, j in sel[np.random.choice(sel.shape[0], num_to_sel, replace=False)]:
                        li.append([k, i, j])

            if False:
                if not (easy.shape[0] > 0 and mid.shape[0] > 0 and hard.shape[0] > 0):
                    continue
                for li, typ in zip([easy_li, mid_li, hard_li], ['easy', 'mid', 'hard']):
                    _, i, j = li[-1]
                    print(li[-1])
                    print(mat[i, k], mat[j, k])

                    voxel_set = set(np.nonzero((mapping[i] > 0) | (mapping[j] > 0))[0].tolist())
                    def elewise_arr_in_set(ele, s):
                        return ele in s
                    arr_in_set = np.vectorize(lambda arr: elewise_arr_in_set(arr, voxel_set))

                    mask = arr_in_set(frame_voxel_idx[k])
                    print(mat[i, k] + mat[j, k])
                    print(mask.mean())

                    img_i = np.array(Image.open(f'ReplicaGen/scene{sid:02d}/rgb/{i:03d}.png').convert('RGBA'))
                    img_j = np.array(Image.open(f'ReplicaGen/scene{sid:02d}/rgb/{j:03d}.png').convert('RGBA'))
                    img_k = np.array(Image.open(f'ReplicaGen/scene{sid:02d}/rgb/{k:03d}.png').convert('RGBA'))
                    img_k[mask, -1] = 160
                    img = np.hstack([img_i, img_k, img_j])

                    Image.fromarray(img).save(f'scene{sid:02d}_{k:03d}_{i:03d}_{j:03d}_{typ}.png')
                    input('press')

        # nums = np.stack(nums)
        # print(sid)
        # print((nums[:, 0] >= 5).sum(), end=' ')
        # print((nums[:, 0] >= 5).sum(), end=' ')
        # print((nums[:, 0] >= 5).sum())

        easy_li = np.asarray(easy_li, np.int32)
        mid_li = np.asarray(mid_li, np.int32)
        hard_li = np.asarray(hard_li, np.int32)

        np.savez_compressed(f'ReplicaGenRelation/scene{sid:02d}_triplet_idx.npz', easy=easy_li, mid=mid_li, hard=hard_li)
