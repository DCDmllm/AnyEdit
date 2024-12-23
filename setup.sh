# python environmet
conda create --name anyedit python=3.10
conda activate anyedit
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu118
python -m spacy download en_core_web_sm  # or manully download the link and pip install en_core_web_sm-3.7.1-py3-none-any.whl

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ..

git clone https://github.com/facebookresearch/segment-anything.git
cd segment_anything
pip install -e .
cd ..

# check path
PYTHONPATH='./' python checkpoints/checkpath.py

# model
export HF_ENDPOINT=https://hf-mirror.com
# 1. foundation_setting for editing
cd checkpoints/foundation_models

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

huggingface-cli download --resume-download bert-base-uncased

cd stable-diffusion-v1-5
huggingface-cli download pt-sk/stable-diffusion-1.5 --local-dir . --local-dir-use-symlinks False
cd ..
wget https://civitai.com/api/download/models/717109?type=Model&format=SafeTensor&size=pruned&fp=fp16 -O realDream_15SD15.safetensors

cd ../stable-diffusion-xl-base-1.0
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir . --local-dir-use-symlinks False

cd ../omost-llama-3-8b-4bits
huggingface-cli download lllyasviel/omost-llama-3-8b-4bits --local-dir . --local-dir-use-symlinks False  # composition
cd ../RealVisXL_V4
huggingface-cli download SG161222/RealVisXL_V4.0 --local-dir . --local-dir-use-symlinks False  # composition

cd ../instruct-pix2pix
huggingface-cli download timbrooks/instruct-pix2pix --local-dir . --local-dir-use-symlinks False # /huggingface/instruct-pix2pix

cd ../ultraedit
huggingface-cli download BleachNick/SD3_UltraEdit_w_mask --local-dir . --local-dir-use-symlinks False

cd ../flux
huggingface-cli download black-forest-labs/FLUX.1-schnell --local-dir . --local-dir-use-symlinks False

cd ../groundingDINO
huggingface-cli download ShilongLiu/GroundingDINO --include groundingdino_swinb_cogcoor.pth --local-dir . --local-dir-use-symlinks False

cd ../sd-inpaint
huggingface-cli download runwayml/stable-diffusion-inpainting --local-dir . --local-dir-use-symlinks False

# 2. material_transfer
cd ../../visual_models/ip_adapter
huggingface-cli download h94/IP-Adapter --local-dir . --local-dir-use-symlinks False

cd ../controlnet-depth-sdxl-1.0
huggingface-cli download diffusers/controlnet-depth-sdxl-1.0 --local-dir . --local-dir-use-symlinks False

cd ../controlnet-canny-sdxl-1.0
huggingface-cli download diffusers/controlnet-canny-sdxl-1.0 --local-dir . --local-dir-use-symlinks False

cd ../stable-diffusion-2-1
huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir . --local-dir-use-symlinks False

cd ..
wget https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt

# 3. visual_image_reference
cd anydoor
wget https://huggingface.co/spaces/xichenhku/AnyDoor/resolve/main/epoch%3D1-step%3D8687.ckpt -O anydoor.ckpt
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
cd ..
wget https://huggingface.co/Fantasy-Studio/Paint-by-Example/resolve/main/model.ckpt -O pae_model.ckpt
huggingface-cli download lllyasviel/ControlNet --include annotator/* --local-dir . --local-dir-use-symlinks False
huggingface-cli download lllyasviel/Annotators --include ControlNetHED.pth --local-dir . --local-dir-use-symlinks False
cd depth_anything_v2
https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

# 4. fliter model
cd ../flitering_model/blip-t5
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Salesforce/blip2-flan-t5-xl --local-dir . --local-dir-use-symlinks False
cd ../vila-1.5-3b
huggingface-cli download Efficient-Large-Model/VILA1.5-3b --local-dir . --local-dir-use-symlinks False

# others, replace the token to your own huggingface token
cd ../../instruction_models/llama3-8b
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct  --local-dir . --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx --local-dir-use-symlinks False
cd ../../
huggingface-cli download openai/clip-vit-large-patch14
