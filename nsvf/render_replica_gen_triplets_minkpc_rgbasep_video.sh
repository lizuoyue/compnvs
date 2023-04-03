# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ReplicaGenEncFtTriplets/easy"
RES="256x256" # hxw
ARCH="mink_nsvf_rgba_sep_field_base"
SUFFIX="v1_pred"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_last.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
MODELARGS=$(printf "$MODELTEMP" 256 0.0)

# rendering with pre-defined testing trajectory
for SID in "13" # "13" "14" "19" "20" "21" "42"
do
CUDA_VISIBLE_DEVICES=2 python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --load-npz "npz_pred/"$SID \
    --path ${MODEL_PATH} \
    --backbone-name "MinkUNet50" \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose_video/$SID \
    --render-views "0..13500" \
    --test-views "0..13500" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/output_no2d_videohhhhhhhh$SID \
    --render-output-types "color" \
    --log-format "simple" \
    --render-combine-output 
done
#"depth" "voxel" "normal"
