import numpy as np
import os

if __name__ == '__main__':

    triplets_folder = '/home/lzq/lzy/replica_habitat_gen/ReplicaGenRelation'
    triplets = np.load(f'{triplets_folder}/scene42_triplet_idx.npz')['easy']
    for k,i,j in triplets:
        src = f'ReplicaGenFt/all/pose/48_{k:03d}.txt'
        tar = f'ReplicaGenEncFtTriplets/easy/pose/48_{k:03d}_{i:03d}_{j:03d}.txt'
        os.system(f'cp {src} {tar}')

    quit()

    ref = np.loadtxt('ReplicaGenFt/all/pose/42_010.txt')
    inv_ref = np.linalg.inv(ref)

    for i in range(300):
        os.system(f'cp ReplicaGenFt/all/rgb/42_{i:03d}.png ReplicaGenFt/all/rgb/48_{i:03d}.png')
        os.system(f'cp ReplicaGenFt/all/depth/42_{i:03d}.npz ReplicaGenFt/all/depth/48_{i:03d}.npz')

        npz = dict(np.load(f'ReplicaGenFt/all/npz/42_{i:03d}.npz'))
        org_pts = np.concatenate([npz['pts'], np.ones((npz['pts'].shape[0], 1), np.float32)], axis=-1)
        org_pose = np.loadtxt(f'ReplicaGenFt/all/pose/42_{i:03d}.txt')

        new_pts = org_pts.dot(inv_ref.T)[:, :3]
        new_pose = inv_ref.dot(org_pose)

        # for pt1, pt2 in zip(org_pts.dot(np.linalg.inv(org_pose).T), new_pts.dot(np.linalg.inv(new_pose).T)):
        #     print(pt1)
        #     print(pt2)
        #     input()

        npz['pts'] = new_pts
        np.savez_compressed(f'ReplicaGenFt/all/npz/48_{i:03d}.npz', **npz)
        np.savetxt(f'ReplicaGenFt/all/pose/48_{i:03d}.txt', new_pose, '%.18lf')


        



    
    
    
