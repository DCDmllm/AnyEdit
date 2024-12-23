#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/local_pipeline_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/flux" \
    --inpaint-ckpt "./checkpoints/foundation_models/sd-inpaint" \
    --lama-config-file "" \
    --lama-ckpt "" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_background_change_prefiltered.jsonl" \
    --image-path './AnyEdit/original_datasets/coco/train2014'\
    --instruction-type "background_change" \