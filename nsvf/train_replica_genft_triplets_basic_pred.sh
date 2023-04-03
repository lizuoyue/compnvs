# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaGenEncFtTriplets/easy"
RES="256x256" # hxw
ARCH="mink_nsvf_base"
SUFFIX="v1_basic_pred"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
CUDA_VISIBLE_DEVICES=2 python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "glob_ReplicaGenEncFtTriplets_easy_train_npz_full" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --backbone-name "MinkUNet50" \
    --pixel-per-view 8192 \
    --no-preload \
    --load-depth \
    --depth-weight 16.0 \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "glob_ReplicaGenEncFtTriplets_easy_trainval_npz_tiny" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --voxel-size 0.1 \
    --load-npz "npz_basic_pred/" \
    --geo-weight 0.0 \
    --ft-weight 16.0 \
    --voxel-embed-dim 32 \
    --inputs-to-density "emb:6:32" \
    --inputs-to-texture "feat:0:256" \
    --feature-embed-dim 256 \
    --density-embed-dim 256 \
    --texture-embed-dim 256 \
    --feature-layers 3 \
    --texture-layers 1 \
    --fix-field \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 512.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 200000 \
    --lr 0.002 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 2000 --max-update 200000 \
    --virtual-epoch-steps 2000 --save-interval 1 \
    --half-voxel-size-at "999999999" \
    --reduce-step-size-at "999999999" \
    --pruning-every-steps 999999999 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}

    # --use-octree \