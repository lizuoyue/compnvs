# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ARKitScenesTripletsNSVF/all"
RES="192x256" # hxw
ARCH="mink_nsvf_rgba_sep_field_base"
SUFFIX="v1"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

# start training locally
while true
do
CUDA_VISIBLE_DEVICES=5 python3 train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "ARKitScenesTriplets_train" \
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
    --valid-views "ARKitScenesTriplets_valtrain" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --voxel-size 0.1 \
    --load-npz "npz/" \
    --geo-weight 0.0 \
    --ft-weight 16.0 \
    --voxel-embed-dim 32 \
    --inputs-to-density "emb:6:8" \
    --inputs-to-texture "feat:0:256" \
    --feature-embed-dim 256 \
    --density-embed-dim 256 \
    --texture-embed-dim 256 \
    --feature-layers 3 \
    --texture-layers 3 \
    --fix-field \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 512.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 300000 \
    --lr 0.0005 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 2000 --max-update 300000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at "999999999" \
    --reduce-step-size-at "999999999" \
    --pruning-every-steps 999999999 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}
done
# --use-octree \
