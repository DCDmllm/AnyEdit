#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/material_transfer_tool.py \
    --groundingdino-config-file "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py" \
    --groundingdino-ckpt "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth" \
    --sam-ckpt "./checkpoints/foundation_models/sam_vit_h_4b8939.pth" \
    --sdxl-ckpt "./checkpoints/foundation_models/stable-diffusion-xl-base-1.0" \
    --image-encoder-path "./checkpoints/foundation_models/ip_adapter/models/image_encoder" \
    --ip-ckpt "./checkpoints/foundation_models/ip_adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin" \
    --controlnet-path "./checkpoints/visual_models/controlnet-depth-sdxl-1.0" \
    --dpt-model-path "./checkpoints/foundation_models/dpt_hybrid-midas-501f0c75.pt" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/coco_material_transfer_prefiltered.json" \
    --image-path "/home1/yqf/datasets/coco/train2014" \
    --instruction-type "visual_material_transfer"