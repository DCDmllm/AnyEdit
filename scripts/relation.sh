#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python relation_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --instruction-path "./AnyEdit/data" \
    --json-path './AnyEdit/instructions/relation_prefilter.json' \
    --image-path './AnyEdit/original_datasets'\
    --instruction-type "relation" \