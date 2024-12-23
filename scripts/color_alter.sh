#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/global_pipeline_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/flux" \
    --ip2p-ckpt "./checkpoints/foundation_models/instruct-pix2pix" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_color_alter_prefiltered.jsonl" \
    --image-path './AnyEdit/original_datasets/coco/train2014'\
    --instruction-type "color_alter" \

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/attribute_pipeline_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/flux" \
    --sd3-ckpt "./checkpoints/foundation_models/ultraedit" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_color_alter_prefiltered.jsonl" \
    --image-path './AnyEdit/original_datasets/coco/train2014'\
    --instruction-type 'color_alter' \