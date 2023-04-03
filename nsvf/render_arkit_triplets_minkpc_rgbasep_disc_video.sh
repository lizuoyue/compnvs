# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="ARKitScenesTripletsNSVF/all"
RES="384x512" # hxw 192x256
ARCH="mink_nsvf_rgba_sep_field_base"
SUFFIX="v1_disc_pred"
DATASET=/home/lzq/lzy/NSVF/${DATA}
SAVE=/home/lzq/lzy/NSVF/${DATA}
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint19.pt # checkpoint_last.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"use_octree":True}'
MODELARGS=$(printf "$MODELTEMP" 256 0.0)

# rendering with pre-defined testing trajectory
CUDA_VISIBLE_DEVICES=9 python3 render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --load-npz "npz_pred_valonly/" \
    --path ${MODEL_PATH} \
    --backbone-name "MinkUNet50" \
    --use-discriminator \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose_video_valonly/"*" \
    --render-views "13920..13935" \
    --test-views "13920..13935" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/hiresoutput_video_duplicate \
    --render-output-types "color" \
    --log-format "simple" \
    --render-combine-output 

# ARKitScenesTriplets_newval
# rebuttal 681 928