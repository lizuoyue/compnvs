import numpy as np
import os

embedding_dir = 'models/vqvae2/embeddings'
scene_embedding_dir = 'models/vqvae2/scene_embeddings'

for phase in ['train', 'val']:
    src = np.load(os.path.join(scene_embedding_dir, f'replica_{phase}.npz'))
    # print(type(src), src['scene_id'].shape, src['embeddings'].shape)
    embeddings = src['embeddings']  # num_scene x 300 x 32 x 32
    num_scenes = embeddings.shape[0]
    new_embeddings = np.concatenate(np.split(embeddings, num_scenes, axis=0), axis=1).squeeze(0)
    print(new_embeddings.shape)
    np.save(os.path.join(embedding_dir, f'replica_{phase}.npy'), new_embeddings)
    print('saved')