import random
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import os
from termcolor import cprint
import yaml
from anyedit.utils import convert_to_np, tokenize_captions
import contextlib

def collate_fn_ip2pSD15(examples):
    original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    if "reference_clip_image" in examples[0].keys():
        reference_clip_images = torch.cat([example["reference_clip_image"] for example in examples], dim=0)
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "reference_clip_images":reference_clip_images,
            "input_ids": input_ids,
            "edit_code": [example["edit_code"] for example in examples]  # codebook number
        }
    else:
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
        }


class AnyEditMixtureDatasetStageIIsd15(torch.utils.data.Dataset):
    def __init__(self, args, tokenizers, accelerator, task_book):
        self.args = args
        self.tokenizers=tokenizers
        self.accelerator=accelerator
        self.train_transforms = transforms.Compose(
            [
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            ]
        )

        self.examples = self._shuffle_and_combine(args=args, accelerator=accelerator, batch_size=8)
        self.clip_image_processor = CLIPImageProcessor()
        self.task_embs_book = task_book

    def _shuffle_and_combine(self, args, accelerator, batch_size):
        print('[INFO] Begin loading dataset')
        all_samples = []

        if accelerator is not None:
            context_manager = accelerator.main_process_first()
        else:
            # Dummy context manager when accelerator is not used
            context_manager = contextlib.nullcontext()

        with context_manager:
            with open(args.yaml_file, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
            assert len(yaml_data['data'].items()) <= batch_size, f"{len(yaml_data['data'].items())} must be smaller than {batch_size}"

            for key, value in yaml_data['data'].items():
                json_file = value['json_file']
                sample_ratio = value['sample_ratio']
                json_path = os.path.join(args.data_root_path, json_file)
                with open(json_path, 'r') as f:
                    dataset = json.load(f)
                num_samples = int(len(dataset) * sample_ratio)
                if sample_ratio > 1:
                    # If sample_ratio > 1, repeat the dataset enough times to satisfy the ratio
                    repeat_count = int(sample_ratio)
                    extended_dataset = dataset * repeat_count
                    remaining_samples = int(len(dataset) * (sample_ratio - repeat_count))
                    extended_dataset.extend(random.sample(dataset, remaining_samples))
                    dataset = extended_dataset
                    num_samples = len(dataset)
                random.seed(args.seed)  # Ensure reproducibility
                sampled_data = random.sample(dataset, num_samples)
                all_samples.extend(sampled_data)

            if hasattr(args, 'max_train_samples') and args.max_train_samples is not None:
                all_samples = all_samples[:args.max_train_samples]

        # shuffle and scatter the image to GPU let The code for even distribution of different types is omitted, we will publish it at the end

        if hasattr(args, 'dataset_repeat_num') and args.dataset_repeat_num is not None:
            all_samples = all_samples * args.dataset_repeat_num
        print('[INFO] End loading dataset')
        return all_samples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        text = item["edit"]
        # if item["edit_type"] == "visual_reference":
        #     text = text.replace("[V*]", "the object in the reference image")

        original_image_path = os.path.join(self.args.data_root_path, item["image_file"])
        edited_image_path = os.path.join(self.args.data_root_path, item["edited_file"])

        # ---------------------- Load images
        original_image = convert_to_np(original_image_path, self.args.resolution)
        edited_image = convert_to_np(edited_image_path, self.args.resolution)
        images = np.concatenate([original_image, edited_image])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        images = self.train_transforms(images)
        # --------------------------

        # Separate images
        original_image_transformed, edited_image_transformed = images.chunk(2)
        original_image_transformed = original_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        edited_image_transformed = edited_image_transformed.reshape(
            3, self.args.resolution, self.args.resolution
        )
        prompt_embeds = tokenize_captions(captions=[text], tokenizer=self.tokenizers)

        if "visual_input" in item.keys() and item["visual_input"] is not None:
            reference_image_path = item["visual_input"].replace(
                "edit_generated_datasets/jellycat/jellycat_images", self.args.data_root_path+"/visual/visual_reference/visual_img"
            )
            reference_raw_image = Image.open(os.path.join(self.args.data_root_path, reference_image_path))
            reference_clip_image = self.clip_image_processor(images=reference_raw_image, return_tensors="pt").pixel_values
        else:
            reference_clip_image = torch.zeros([1, 3, 224, 224])

        if "edit_type" in item.keys() and item["edit_type"] is not None:
            edit_type = item["edit_type"]
        else:
            edit_type = None

        return {
            "original_pixel_values": original_image_transformed,
            "edited_pixel_values": edited_image_transformed,
            "reference_clip_image": reference_clip_image,
            "input_ids": prompt_embeds[0],
            "edit_code": self.task_embs_book[edit_type]
        }