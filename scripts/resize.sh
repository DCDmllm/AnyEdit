#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python  AnyEdit_Collection/adaptive_editing_pipelines/move_resize_pipeline_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/stable-diffusion-xl-base-1.0" \
    --lama-config-file "./lama/configs/prediction/default.yaml" \
    --lama-ckpt "./checkpoints/foundation_models/big-lama" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_movement_prefiltered.jsonl" \
    --image-path './AnyEdit/original_datasets/coco/train2014'\
    --instruction-type "movement" \