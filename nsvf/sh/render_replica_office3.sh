# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="Replica/office3"
RES="256x256" # hxw
ARCH="nsvf_base"
SUFFIX="v1"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}_init_voxel
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_best.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
MODELARGS=$(printf "$MODELTEMP" 256 0.0)

# rendering with pre-defined testing trajectory
CUDA_VISIBLE_DEVICES=1 python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "0..10" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"
