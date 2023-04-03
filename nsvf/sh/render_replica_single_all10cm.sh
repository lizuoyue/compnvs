# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaSingleAll10cm/all"
RES="256x256" # hxw
ARCH="scn_nsvf_base"
SUFFIX="v1"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_last.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
MODELARGS=$(printf "$MODELTEMP" 256 0.0)

# rendering with pre-defined testing trajectory
CUDA_VISIBLE_DEVICES=6 python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --load-npz \
    --path ${MODEL_PATH} \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "0..10" \
    --test-views "0..10" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --log-format "simple" \
    --render-combine-output 
