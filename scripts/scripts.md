# Examples

```bash
bash [pipeline].sh [gpu_id]
```

## Text Input

#### pipeline (1)

```bash
bash scripts/add.sh 0
bash scripts/remove.sh 0
bash scripts/counting.sh 0
bash scripts/background_change.sh 0
bash scripts/replace.sh 0
```

#### pipeline (2) 

```bash
bash scripts/appearance_alter.sh 0,1
bash scripts/color_alter.sh 0
bash scripts/tone_transfer.sh 0
```

#### pipeline (3)

```bash
bash scripts/action_change.sh 0  # it can also change some fine-grained actions (e.g. emotional expression change)
```

#### pipeline (4)
```bash
bash scripts/resize.sh 0
bash scripts/movement.sh 0
```

#### pipeline (5)
```bash
bash scripts/implicit_change 0,1
```

#### pipeline (6)
```bash
bash scripts/relation 0
```

#### pipeline (7)
```bash
bash scripts/visual_image_reference.sh 0
```

#### pipeline (8)
1. Download our material image in [FMD](https://people.csail.mit.edu/lavanya/fmd.html) and [KTH_TIPS](https://paperswithcode.com/dataset/kth-tips2)
```bash
# need to put all material image into the path './AnyEdit/data/visual_material_transfer/materials'
bash scripts/visual_material_transfer.sh 0
```

## Others

#### Rotation Change (9)
1. Download [MVImageNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet) following their link, Please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfU9BkV1hY3r75n5rc37IvlzaK2VFYbdsvohqPGAjb2YWIbUg/viewform?usp=sf_link) to get the download link and password, then unzip 
2. Folder structure
```shell
|-- ROOT
    |-- class_label
        |-- instance_id
            |-- images
            |-- sparse/0
                |-- cameras.bin   # COLMAP reconstructed cameras
                |-- images.bin    # binary data of input images
                |-- points3D.bin  # COLMAP reconstructed sparse point cloud (not dense) 
```
3. Update the parameters
```bash
cd ROOT/class_label/instance_id
mkdir text_annotations
```
- Replace '--input_model' to the path of 'sparse/0', '--output_model' to the path of 'text_annotations'
- Replace '--text_file' to the path of 'xxx/text_annotations/image.txt', '--image_root' to the path of 'xxx/images'
- Update the category of this image (MVImageNet contain 180 categories)
```bash
bash scripts/rotation_change.sh 0
```

#### Outpaint (10)

```bash
bash scripts/outpainting.sh 0
```

#### Textual Change (11)

use FLUX

```bash
bash scripts/textual_change.sh 0
```

#### other visual input (12)

(scribble, segmentation, depth, canny, boundingbox(mask))

Refer to 
```
AnyEdit_Collection/adaptive_editing_pipelines/visual_condition_tool.ipynb
```

#### Style Transfer (13)
Please refer to this [prisma](https://prisma-ai.com/).

#### Material Change (14)
change "the material of [object] like the image [V*]" in **Material Transfer** to "change the material of [object] to [material category]", where [material category] can obtained from instruction['material']

