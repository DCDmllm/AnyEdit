#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/textual_change_tool.py