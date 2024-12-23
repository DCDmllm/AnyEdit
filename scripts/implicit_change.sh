export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/implicit_tool.py \
    --base-model-path "./checkpoints/foundation_models/stable-diffusion-xl-base-1.0" \
    --image-encoder-path "./checkpoints/foundation_models/ip_adapter/sdxl_models/image_encoder" \
    --ip-ckpt "./checkpoints/foundation_models/ip_adapter/sdxl_models/ip-adapter_sdxl.bin" \
    --controlnetXL-ckpt "./checkpoints/visual_models/controlnet-canny-sdxl-1.0" \
    --instruction-path "./AnyEdit/data" \
    --json-path "./AnyEdit/instructions/mscoco_implicit_change_prefiltered.jsonl" \
    --instruction-type "implicit_change" \