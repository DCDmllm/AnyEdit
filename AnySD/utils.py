import numpy as np
import PIL
import torch

task_embs_book = {
    "action_change": 0,
    "material_change": 1,
    "remove": 2,
    "style_change": 3,
    "visual_depth": 4,
    "visual_segment": 5,
    "add": 6,
    "color_alter": 7,
    "movement": 8,
    "replace": 9,
    "textual_change": 10,
    "visual_mat": 11,
    "visual_sketch": 12,
    "appearance_alter": 13,
    "counting": 14,
    "outpainting": 15,
    "resize": 16,
    "tone_transfer": 17,
    "visual_ref": 18,
    "background_change": 19,
    "implicit_change": 20,
    "relation": 21,
    "rotation_change": 22,
    "visual_bbox": 23,
    "visual_scribble": 24
}

trival_embs_book = {key: 0 for key in task_embs_book}

# SDXL
# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# SDXL
# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompts(text_encoders, tokenizers, prompts):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)

    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

# SDXL
# Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers, accelerator):
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(text_encoders, tokenizers, prompts)
        add_text_embeds_all = pooled_prompt_embeds_all

        prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
        add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
    return prompt_embeds_all, add_text_embeds_all

# SD-1.5
def tokenize_captions(tokenizer, captions):
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def print_format_tree(obj, indent=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{indent}{key}:")
            print_format_tree(value, indent + "  ")
    elif isinstance(obj, torch.Tensor):
        print(f"{indent}Tensor of shape {list(obj.shape)}")
    else:
        print(f"{indent}{type(obj)}")