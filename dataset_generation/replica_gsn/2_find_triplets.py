import numpy as np
from PIL import Image

if __name__ == '__main__':

    window_size = 10
    eps = 0.5
    result = []

    relations = np.load('train_relations.npz')['relations']
    for sid in range(relations.shape[0]):
        mat = relations[sid]
        for k in range(100):
            valid = (0 < mat[:, k]) & (mat[:, k] < eps)
            valid = np.nonzero(valid.flatten())[0]
            if valid.shape[0] < 2:
                pass
            else:
                for u in range(valid.shape[0] - 1):
                    i = valid[u]
                    for v in range(i + 1, valid.shape[0]):
                        j = valid[v]
                        if k - window_size < i and i < k and k < j and j < k + window_size and mat[i, j] == 0 and mat[j, i] == 0:
                            if False:
                                img_i = np.array(Image.open(f'ReplicaGSN/scene{sid:03d}/rgb/{i:02d}.png'))
                                img_j = np.array(Image.open(f'ReplicaGSN/scene{sid:03d}/rgb/{j:02d}.png'))
                                img_k = np.array(Image.open(f'ReplicaGSN/scene{sid:03d}/rgb/{k:02d}.png'))
                                img = np.hstack([img_i, img_j, img_k])[..., :3]
                                Image.fromarray(img).save(f'scene{sid:03d}_{i:02d}_{j:02d}_{k:02d}.jpg')
                                input('press')
                            result.append((i, j, k))
    
    print(len(result))
