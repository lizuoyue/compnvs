# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaGen"
RES="256x256" # hxw
ARCH="reg_multi_scene_nsvf_base"
SUFFIX="v1_dim64emb64"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
CUDA_VISIBLE_DEVICES=8 python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..290" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --dim-compressed 64 \
    --voxel-embed-dim 64 \
    --inputs-to-density "emb:6:64" \
    --no-preload \
    --load-depth \
    --depth-weight 32.0 \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "285..300" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --voxel-path ${DATASET}/init_voxel.txt \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 128.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 300000 \
    --lr 0.001 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 5000 --max-update 400000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "9999999,99999999" \
    --reduce-step-size-at "9999999,99999999" \
    --pruning-every-steps 99999999 \
    --pruning-th 0.5 \
    --voxel-size 0.1 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}
