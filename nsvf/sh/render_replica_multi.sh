# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaMultiScene"
RES="256x256" # hxw
ARCH="multi_scene_nsvf_base"
SUFFIX="v1_10cm_refine"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_best.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
MODELARGS=$(printf "$MODELTEMP" 256 0.0)

# rendering with pre-defined testing trajectory
CUDA_VISIBLE_DEVICES=4 python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/scene00/pose \
    --render-views "240..250" \
    --test-views "240..250" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --log-format "simple" \
    --render-combine-output 
