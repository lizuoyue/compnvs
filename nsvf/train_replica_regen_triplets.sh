# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaRegenTriplets/all"
RES="256x256" # hxw
ARCH="geo_scn_nsvf_base"
SUFFIX="v1"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
CUDA_VISIBLE_DEVICES=4 python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "gen_train" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 1 \
    --pixel-per-view 8192 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "gen_train_val_small_random" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --voxel-size 0.1 \
    --load-npz \
    --fix-field \
    --geo-weight 0.0 \
    --ft-weight 1.0 \
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
    --save-interval-updates 1000 --max-update 300000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at "999999999" \
    --reduce-step-size-at "999999999" \
    --pruning-every-steps 999999999 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}

    # --use-octree \
