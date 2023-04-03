1. Train VQ-VAE:

2. Extract embeddings on trained VQ-VAE:

3. Extract orders for outpainting according to masks:

4. Train outpaint models along with refinement models on the extracted embeddings and outpainting orders:

5. Run outpaint + refinement on validation set:

6. Train depth estimation model:

7. Run depth estimation on validation set:
# CUDA_VISIBLE_DEVICES=3 python run_depth_my.py --load_params_from models/depth/checkpoints/replica/UNet_36.pt --dataset replica_mid