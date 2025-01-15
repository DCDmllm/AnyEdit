# Diverse Instruction Generation

## File Tree

```smail
└── diverse_Instruction_generation
    ├── edit_instruction  # for other instructions
    ├── implicit # for 'implicit change' instructions 
    └── captions_generator.py # for AnyEdit-Composition
```

## Pipeline


### Prepare Environment

```shell
bash setup.sh
```

### Download the original datasets
| Dataset name | Source                        |
| ------------------------------------------------------------ | ----------------------------------- |
| COCO    |      https://cocodataset.org/#download   |  
|LLaVA-CC3M-Pretrain  | https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K |
| MVImgNet | https://github.com/GAP-LAB-CUHK-SZ/MVImgNet |
| AnyEdit-Composition | we provide our final filtered [concept.json](AnyEdit_Collection/diverse_Instruction_generation/concept/concept.json) and run *captions_generator.py* with 'cc2cap' to generation the corresponding captions.

### Instruction Generation

For most edit types, see the following scripts to generate instructions.

For visual reference，run *visual_reference_crawler.py* to obtain corresponding visual reference images，and save into *./AnyEdit/data/visual_reference/visual_input*.

For visual_material_transfer/material_change, download material images from [FMD](https://people.csail.mit.edu/lavanya/fmd.html) and [KTH_TIPS](https://www.csc.kth.se/cvap/databases/kth-tips/index.html) with their categories. Then transfer the instruction of replace to visual_material_transfer (e.g., *change the material of [objects] to [V\*]/Material Category*) to guarantee the effectiveness of edited objects.

For style change, we direct use the style category to construct instructions (e.g., *change the style of the image to Style Category*).

For textual change, download [AnyWord](https://modelscope.cn/datasets/iic/AnyWord-3M/files), then rename the *ocr_data/Art/data.json* to *textual_art.json*, and so on. 

For visual condition (bbox/depth/scribble/segment/sketh), we transfer the instruction of **replace** and **remove** to visual condition instructions. Specically, the instruction is like *Watch/Refer/Follow the given image to remove the [objects]* or *Watch/Refer/Follow the given image to replace the [objects] to [new objects]*. We randomly choose one for the visual condition instruction.

```shell
# "add", 'remove', "replace", "color_alter", "action_change", "background_change", "tone_transfer", "appearance_alter"
PYTHONPATH='./' python AnyEdit_Collection/diverse_Instruction_generation/instruction_gen.py --instruction-type xxx --gpu-id 0 --batch-size 6 --source-data mscoco
# "textual_change"
PYTHONPATH='./' python AnyEdit_Collection/diverse_Instruction_generation/instruction_gen.py --instruction-type textual --gpu-id 0 --batch-size 6 --source-data mscoco
# "counting", "resize", "movement", "relation", "visual_reference"
PYTHONPATH='./' python AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/other_instruction_gen.py --target-path xxxx --edit-type xxxx --source-data mscoco
# "implicit_change"
# transform the text prompt provided by GPT into dict and text_gen_full.json to Anyedit
PYTHONPATH='./' python implicit/instruction_gen.py
PYTHONPATH='./' python implicit/deal_text2json.py
# "outpaint"
# it generate instructions and edit results together.
PYTHONPATH='./' python AnyEdit_Collection/adaptive_editing_pipelines/outpainting.py
```

# 参考

[1] Synth2: Boosting Visual-Language Models with Synthetic Captions and Image Embeddings object

[2] MetaCLIP

[3] SynCLR

[4] SynthCLIP
