#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1


# example
BASE_DIR='./AnyEdit/original_datasets/MVImgNet/8'
OUTPUT_ROOT='./AnyEdit/data/rotation_change/8_sofa'

for SUB_DIR in $(find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type d); do
    echo "Processing directory: ${SUB_DIR}"

    PYTHONPATH='./' python read_write_camera_model.py \
        --input_model "${SUB_DaIR}/sparse/0" \
        --input_format '.bin' \
        --output_model "${SUB_DIR}/text_annotations" \
        --output_format '.txt'

    echo "----read_write_camera_model done for ${SUB_DIR}----"
done

# -m debugpy --listen 60466 --wait-for-client 
PYTHONPATH='./' python  AnyEdit_Collection/adaptive_editing_pipelines/rotation_change_tool.py \
    --text_file "text_annotations/images.txt" \
    --image_root "images" \
    --root_dir "${BASE_DIR}" \
    --output_dir "${OUTPUT_ROOT}"\
    --category sofa

