#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/action_change_tool.py \
  --model-path "./checkpoints/foundation_models/realDream_15SD15.safetensors" \
  --instruction-path './AnyEdit/data' \
  --instruction-type "action_change" \
  --json-path "./AnyEdit/instructions/mscoco_action_change_prefiltered.jsonl" \