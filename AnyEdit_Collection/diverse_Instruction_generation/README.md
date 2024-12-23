# Diverse Instruction Generation

## File Tree

```smail
└── InstructionGen
    ├── concept
    │   ├── edit.py
    │   └── xxx.py
    ├── checkpoints
    │   ├── llama3-8b (recommond)
    │   └── mistral-7b
    └── captions_generator.py
```

## Pipeline

| ID   | **Type**                 | **Details**                                                  |
| ---- | ------------------------ | ------------------------------------------------------------ |
| 1    | 1 concept + 1 background | 1. Generate about 5 background for  each concept 2. give instruction for each 1 concept + 1 background |
| 2    | 2 concept composition    |                                                              |
| 3    | Anyedit Instruction      |                                                              |
| 4    | Implicit                 |                                                              |

## Instruction Generation

### Download the concept pool (from scratch)

| version                                                      | description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [classname.json ](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 207个[图片分类数据](https://paperswithcode.com/datasets?task=image-classification)收集得到的 23,028 个概念， 主要来自[CIFAR10](https://paperswithcode.com/dataset/cifar-10),  [CIFAR100](https://paperswithcode.com/dataset/cifar-100), [ImageNet1k](),[CUB-200](https://paperswithcode.com/dataset/cub-200-2011),[Oxford 102 Flower](https://paperswithcode.com/dataset/oxford-102-flower), [STL10](https://paperswithcode.com/dataset/stl-10), [food101](https://paperswithcode.com/dataset/food-101), [sun397](https://paperswithcode.com/dataset/sun397), [Sports10](https://paperswithcode.com/dataset/sports10),[CNFOOD-241](https://data.mendeley.com/datasets/fspyss5zbb/1), [ifood251](https://github.com/karansikka1/iFood_2019), [Grocery Store](https://github.com/marcusklasson/GroceryStoreDataset) , [voc2007](https://paperswithcode.com/dataset/pascal-voc-2007), [Oxford Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [resisc45](https://paperswithcode.com/dataset/resisc45), [kinetics700](https://paperswithcode.com/dataset/kinetics-700), [ucf101](https://www.crcv.ucf.edu/data/UCF101.php)，[openimageV7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html#class-definitions), [TecentMI](https://github.com/Tencent/tencent-ml-images/blob/master/data/dictionary_and_semantic_hierarchy.txt), [iCartoonFace](https://github.com/luxiangju-PersonAI/iCartoonFace?tab=readme-ov-file#Dataset) |
| [SynCLR_concept+background.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM)   [SynCLR_concept.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 使用[SynCLR](https://github.com/google-research/syn-rep-learn/tree/main/SynCLR/synthesis/syn_text)中的`imgnet21k_combined_background_dict`和`imgnet_background_dict`得到 $13,890$ 个概念, 每个concept配了约100个background |
| [concept.json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM) | 上面两个文件合成的$35,299$个概念                             |
| [concept_pool(2409 version).json](https://drive.google.com/drive/folders/17fSf4Dr-_lfTHIt5_xnPzBsg3nr8XKOM?ths=true) | 人工筛完的concept                                            |
| [**concept.json**](./concept/concept.json)                                                      | GPT再筛选掉得到最终的100个concept以及对100个尾部background                                           |
|                                                              |                                                              |

Finally, we provide the filtered 100 tail concepts and 100 tail backgrounds [**concept.json**](./concept/concept.json) for **Counterfactual Synthetic Scene Pair Dataset**, which contain 22936 samples and 500 compositional concepts

## 1,2 Caption Generation


```shell
python captions_generator.py --gpu-id 0 --mode cc2cap # Counterfactual Synthetic Scene Pair Dataset
```

## 3 Edit Instruction

| dataset name                                                 | support type                        |
| ------------------------------------------------------------ | ----------------------------------- |
| [COCO](), [llava595]()                                          | add, remove, replace, action_change, etc |
| [AnyWord](https://modelscope.cn/datasets/iic/AnyWord-3M/files)(Art, ): Rename *ocr_data/Art/data.json* to *textual_art.json* | textual (limitation: only supporting English)   |
|                                                              |                                     |

- Download [anyedit_original_captions.zip](https://drive.google.com/file/d/1TPMhKQMabXLaNtnDxuJzT5TgdrPUuH0m/view?usp=drive_link), unzip the file and store these original captions into the *./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input* folder for diverse instruction generation.
- The *source-data* can selected from ["mscoco", "llava595", "art", "coco_text", 'icdar2017rctw', 'LSVT', 'mlt2019', 'MTWI2018', 'ReCTS']
```shell
# single turn + batch (example)
PYTHONPATH='./' python edit_instruction/instruction_gen.py --instruction-type textual --gpu-id 6 --batch-size 6 --source-data icdar2017rctw
# multi_turn
PYTHONPATH='./' python edit_instruction/instruction_gen_multi_turn.py --instruction-type mix --gpu-id 4 --iter 3
```

## 4 Implict

```bash
# generate instruction
PYTHONPATH='./' python implicit/instruction_gen.py
# transform the text prompt provided by GPT into dict
PYTHONPATH='./' python implicit/deal_text2json.py
# text_gen_full.json to Anyeidt and run t2i
```

# References

[1] Synth2: Boosting Visual-Language Models with Synthetic Captions and Image Embeddings object

[2] MetaCLIP

[3] SynCLR

[4] [SynthCLIP](https://github.com/hammoudhasan/SynthCLIP/tree/main/TextGen)
