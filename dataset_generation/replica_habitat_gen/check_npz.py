import numpy as np
import glob
import os
from PIL import Image

def insert(d, key, val):
    if key in d:
        d[key].add(val)
    else:
        d[key] = {val}
    return

if __name__ == '__main__':

    for sid in [13, 14, 19, 20, 21, 42]:

        files = sorted(glob.glob(f'ReplicaGenEncFtTriplets/easy/npz_pred/{sid:02d}*'))
        files = [[int(item) for item in os.path.basename(file).replace('.npz', '').split('_')] for file in files]

        ij2k, ikj2img, relation = {}, {}, {}
        for fid, (_, k, i, j) in enumerate(files):
            imgid = fid * 15
            insert(relation, i, j)
            insert(relation, j, i)
            insert(ij2k, (i, j), k)
            insert(ij2k, (j, i), k)
            ikj2img[(i, k, j)] = [f'ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output{sid:02d}/color/{x:05d}.png' for x in range(imgid, imgid + 15)]
            ikj2img[(j, k, i)] = ikj2img[(i, k, j)][::-1]
        
        for i, j in ij2k:
            inter = list(relation[i].intersection(relation[j]))
            if len(inter) > 0:
                for n in inter:
                    for kij in ij2k[(i, j)]:
                        for kin in ij2k[(i, n)]:
                            for kjn in ij2k[(j, n)]:
                                ims = ikj2img[(i, kij, j)] + ikj2img[(j, kjn, n)][1:] + ikj2img[(n, kin, i)][1:-1]
                                ims = [Image.open(im) for im in ims]
                                ims[0].save(f'multiview_gif/{sid:02d}_{i:03d}_{kij:03d}_{j:03d}_{kjn:03d}_{n:03d}_{kin:03d}.gif',append_images=ims[1:],duration=100,loop=0,save_all=True)

    if False:

        files = sorted(glob.glob(f'/home/lzq/lzy/NSVF/ARKitScenesTripletsNSVF/all/npz_pred/*')) # _valonly
        files = [[int(item) for item in os.path.basename(file).replace('.npz', '').split('_')] for file in files]

        ij2k, relation = {}, {}
        for sid, k, i, j in files:
            ij2k[(sid, i, j)] = k
            ij2k[(sid, j, i)] = k
            insert(relation, (sid, i), j)
            insert(relation, (sid, j), i)
        

        for sid, i, j in ij2k:
            inter = list(relation[(sid, i)].intersection(relation[(sid, j)]))
            if len(inter) > 0:
                for n in inter:
                    print(sid, i, j, n)#, ij2k[(i, j)], ij2k[(i, n)], ij2k[(j, n)])
        
        # for sid, i, j in ij2k:
        #     for n in relation[(sid, i)]:
        #         if n != j:
        #             inter = list(relation[(sid, n)].intersection(relation[(sid, j)]))
        #             assert(i in inter)
        #             inter = [m for m in inter if m != i]
        #             if len(inter) > 0:
        #                 for m in inter:
        #                     print(sid, i, j, n, m)#ij2k[(sid, i, j)], ij2k[(sid, i, n)], ij2k[(sid, j, n)])

        # ReplicaGenEncFtTriplets/easy/mink_nsvf_rgba_sep_field_base/output14/color/00002.png
        


    
    quit()



    a_files = sorted(glob.glob(f'ReplicaGenEncFtTriplets/easy/npz_basic_pred/13*'))
    b_files = sorted(glob.glob(f'ReplicaGenEncFtTriplets/easy/npz_pred/13*'))

    for afile, bfile in zip(a_files, b_files):
        a = np.load(afile)
        b = np.load(bfile)

        for d in [a,b]:
            for key in d:
                print(key, d[key].shape, d[key].dtype)
        input()



        print(list(a.keys()))
        print(list(b.keys()))
        input()
        # for key in a:
        #     print(key, a[key].shape, end=' ')
        #     assert(a[key].shape[0] == b[key].shape[0])
        # print()
        # for key in a:
        #     if len(a[key].shape)>1 and a[key].shape[1] == b[key].shape[1]:
        #         print(key, (a[key]==b[key]).mean(), end=' ')
            
        print()
        print()
        # print()
        # for key in b:
        #     print(b[key].shape, end=' ')

        # print(a['vertex_idx'].shape)
        # print(b['vertex_idx'].shape)
        # input('')

        # print(len(a_files), len(b_files))



    