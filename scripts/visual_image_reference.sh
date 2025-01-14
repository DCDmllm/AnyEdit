#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1 

PYTHONPATH='./' python  AnyEdit_Collection/adaptive_editing_pipelines/visual_reference_tool.py \
    --pretrained_model "./checkpoints/visual_models/anydoor/checkpoints/anydoor.ckpt" \
    --config "./checkpoints/visual_models/anydoor/configs/anydoor.yaml" \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/flux" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_visual_reference_prefiltered.jsonl" \
    --image-path './AnyEdit/original_datasets/coco/train2014'\
    --instruction-type "visual_reference" \