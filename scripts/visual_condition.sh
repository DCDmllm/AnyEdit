#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.py \
    --seg-config "./AnyEdit_Collection/other_modules/uniformer/seg_config.py" \
    --seg-ckpt "./checkpoints/visual_models/annotator/ckpts/upernet_global_small.pth" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_segment_prefiltered.json" \
    --image-path './AnyEdit/original_datasets' \
    --instruction-type "visual_segment" \

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.py \
    --scribble-ckpt "./checkpoints/visual_models/ControlNetHED.pth" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_scribble_prefiltered.json" \
    --image-path './AnyEdit/original_datasets' \
    --instruction-type "visual_scribble" \

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.py \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_sketch_prefiltered.json" \
    --image-path './AnyEdit/original_datasets' \
    --instruction-type "visual_sketch" \

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.py \
    --depth-ckpt "./checkpoints/visual_models/depth_anything_v2/depth_anything_v2_vitl.pth" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_depth_prefiltered.json" \
    --image-path './AnyEdit/original_datasets' \
    --instruction-type "visual_depth" \

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_bbox_prefiltered.json" \
    --image-path './AnyEdit/original_datasets' \
    --instruction-type "visual_bbox" \