'''
ref to https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
'''
import os
import torch
from termcolor import cprint
from anyedit.adapter import Resampler, IPAdapterAttnProAnySD
from diffusers.models.embeddings import MultiIPAdapterImageProjection
from anyedit.adapter.attention_processor import AttnProcessor2_0

class MoE(torch.nn.Module):
    def __init__(self, unet, image_encoder, expert_num, num_tokens=16, ckpt_path=None):
        '''
        num_tokens = 16  # "Number of tokens to query from the CLIP image encoding."
        '''
        super().__init__()

        self.adapter_modules = self._initialize_adapter_modules(unet, num_tokens) # 要在proj前，否则会拷贝进去
        self.image_proj_model = self._initialize_image_proj_model(unet, image_encoder, num_tokens)
        self.unet = unet
        self.task_embs = torch.nn.Parameter(torch.randn(expert_num, 1, 1, 768))  # 先初始化1个看看, 中间2个1是为了更好得匹配
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def _initialize_image_proj_model(self, unet, image_encoder, num_tokens):
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/unet.py#L874C1-L882C59
        from diffusers.models.embeddings import IPAdapterPlusImageProjection
        # image_projection_layer = IPAdapterPlusImageProjection(
        #         embed_dims=image_encoder.config.hidden_size,
        #         output_dims=unet.config.cross_attention_dim,
        #         hidden_dims=unet.config.cross_attention_dim,
        #         heads=12,
        #         depth=4,
        #         ffn_ratio=4,
        #         num_queries=num_tokens,
        # )
        image_projection_layer = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4
        )
        image_projection_layers = [image_projection_layer]

        unet.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        unet.config.encoder_hid_dim_type = "ip_image_proj"

        return image_projection_layer

    def _initialize_adapter_modules(self, unet, num_tokens):
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAdapterAttnProAnySD(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   num_tokens=num_tokens)
                attn_procs[name].load_state_dict(weights)
        unet.set_attn_processor(attn_procs) # set in unet , https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/unet.py#L871
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        return adapter_modules

    def forward(self, concatenated_noisy_latents, timesteps, encoder_hidden_states,
                reference_image_embeds=None, edit_code=None):
        # 可能直接加载第一个CLS token上，这是最简单的实现方式， 可以参考这个看看 https://blog.csdn.net/weixin_44791964/article/details/129941386
        # encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        t_list = []

        if edit_code is None:
            for idx in range(reference_image_embeds.shape[0]):
                t_list.append(self.task_embs[0])
        else:
            for e_code in edit_code:
                t_list.append(self.task_embs[e_code])

        added_cond_kwargs = {"task_embeds": torch.stack(t_list, dim=0)}

        if reference_image_embeds is not None:
            added_cond_kwargs["image_embeds"] = reference_image_embeds

        noise_pred = self.unet(
            concatenated_noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        state_dict = {
            "image_proj": self.image_proj_model.state_dict(),
            "ip_adapter": self.adapter_modules.state_dict()
        }
        ip_sd = {}
        for k in state_dict:
            if k.startswith("unet"):
                pass
            elif k.startswith("image_proj"):
                ip_sd[k.replace("image_proj.", "")] = state_dict[k]
            elif k.startswith("ip_adapter"):
                ip_sd[k.replace("ip_adapter.", "")] = state_dict[k]

        torch.save(ip_sd, os.path.join(output_dir, "ip_adapter_sd15.bin"))  # 可以直接被diffusers调用

        # 保存 task_embs 到一个单独的文件
        torch.save(self.task_embs.detach().cpu(), os.path.join(output_dir, "task_embs.bin"))

    def load_from_checkpoint(self, ckpt_path: str):
        # 用于恢复训练
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class AnyDMPipeline:
    def __init__(self, sd_pipe, image_encoder_path, adapters_list: dict, task_embs_checkpoints):
        self.sd_pipe = sd_pipe
        self.image_encoder_path = image_encoder_path
        self.adapters_list = adapters_list
        self.active_adapter = next(iter(self.adapters_list.keys())) # first key
        self._load_adapter()
        self._load_task_embs(task_embs_checkpoints=task_embs_checkpoints)

    def _load_task_embs(self, task_embs_checkpoints):
        self.sd_pipe.task_embs = torch.nn.Parameter(torch.randn(1, 1, 1, 768))
        self.sd_pipe.task_embs.data = torch.load(task_embs_checkpoints).to(
            device=self.sd_pipe.device, dtype=self.sd_pipe.dtype)   # directly load to sd_pipe

    def _load_adapter(self):
        path = self.adapters_list[self.active_adapter]
        path_split = path.split('/')
        weight_name=path_split[-1]
        subfolder=path_split[-2]
        ckpt_path = path.replace(f"{subfolder}/{weight_name}", "")
        # ref to https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/ip_adapter.py#L55
        self.sd_pipe.load_ip_adapter(ckpt_path, subfolder=subfolder, weight_name=weight_name,
                                     image_encoder_folder=self.image_encoder_path)

    def _shift_adapter(self, name):
        if self.active_adapter != name:
            assert name in self.adapters_list.keys(), f"Adapter '{name}' not loaded."
            self.active_adapter = name
            self._load_adapter()

    def list_adapters(self):
        return list(self.adapters_list.keys())

    def _set_ref_scale(self, scale):
        self.sd_pipe.set_ip_adapter_scale(scale)

    def _process_ref_image_cfg(self, reference_image):
        if reference_image is not None:
            # 其实可以直接输入 ip_adapter_image=reference_image, 但是为了保存训练和推理一致，我就拎出来了
            reference_image_embeds, negative_reference_image_embeds = self.sd_pipe.encode_image(
                reference_image, "cuda", 1, self.sd_pipe.unet.encoder_hid_proj.image_projection_layers
            )
            reference_image_embeds = torch.cat(
                [reference_image_embeds, negative_reference_image_embeds, negative_reference_image_embeds]
            ).unsqueeze(1)
        else:

            reference_image_embeds, negative_reference_image_embeds = self.sd_pipe.encode_image(
                torch.zeros([1, 3, 224, 224]), "cuda", 1, self.sd_pipe.unet.encoder_hid_proj.image_projection_layers
            )
            reference_image_embeds = torch.cat(
                [reference_image_embeds, negative_reference_image_embeds, negative_reference_image_embeds]
            ).unsqueeze(1)

        return reference_image_embeds

    @torch.no_grad()
    def __call__(
        self,
        reference_image=None,
        original_image=None,
        prompt=None,
        negative_prompt=None,
        num_samples=4,
        generator=None,
        guidance_scale=7.5,
        original_image_guidance_scale=2.5,
        reference_image_guidance_scale=0.8,
        num_inference_steps=30,
        adapter_name=None,
        e_code=None,  # task emb code number
    ):
        if adapter_name is not None:
            self._shift_adapter(name=adapter_name)
        self._set_ref_scale(scale=reference_image_guidance_scale)

        images = self.sd_pipe(
            prompt=prompt,
            ip_adapter_image_embeds=[self._process_ref_image_cfg(reference_image)],
            image=original_image,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image_guidance_scale=original_image_guidance_scale,
            e_code=e_code
        ).images

        return images