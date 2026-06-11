"""``SanaSprintHFBackbone`` — Sana-Sprint via HuggingFace diffusers.

Transformer (LoRA-trained, bf16) + AutoencoderDC VAE (frozen, fp32) + Gemma-2 text
encoder + SCMScheduler. Differentiable sampling re-implements the SanaSprintPipeline
denoising loop (trigflow SCM) with gradient flow and optional step checkpointing.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch

from ..base import Backbone


class SanaSprintHFBackbone(Backbone):
    name = "sana_sprint_hf"

    def __init__(
        self,
        *,
        model_repo: str,
        image_size: int = 1024,
        sample_steps: int = 2,
        max_timesteps: float = 1.57080,
        intermediate_timesteps: float = 1.3,
        cfg_scale: float = 4.5,
        weight_dtype: str = "bfloat16",
        vae_dtype: str = "float32",
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[list] = None,
        use_grad_checkpoint: bool = True,
    ) -> None:
        self.model_repo = str(model_repo)
        self.image_size = int(image_size)
        self.sample_steps = int(sample_steps)
        self.max_timesteps = float(max_timesteps)
        self.intermediate_timesteps = None if intermediate_timesteps is None else float(intermediate_timesteps)
        self.cfg_scale = float(cfg_scale)
        self.weight_dtype = getattr(torch, weight_dtype)
        self.vae_dtype = getattr(torch, vae_dtype)
        self.max_sequence_length = int(max_sequence_length)
        # None -> use the pipeline's default chi-prompt.
        self.complex_human_instruction = (
            list(complex_human_instruction)
            if complex_human_instruction is not None else None
        )
        self.use_grad_checkpoint = bool(use_grad_checkpoint)

        # filled by load_models / inject_lora / prepare
        self.model = None
        self._peft_model = None
        self.vae = None
        self.scheduler = None
        self._pipe = None
        self.device = None
        self.sigma_data = None
        self.guidance_scale_embed = None
        self.latent_shape = None

    def load_models(self, device) -> None:
        from diffusers import (
            AutoencoderDC, SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler,
        )
        from transformers import AutoTokenizer, Gemma2Model

        self.device = device

        # Load components explicitly (not via from_pretrained): the auto-loader
        # forwards offload_state_dict into the Gemma2 ctor, which transformers
        # rejects. The pipeline ctor still gives us its chi-prompt encode_prompt.
        tokenizer = AutoTokenizer.from_pretrained(self.model_repo, subfolder="tokenizer")
        text_encoder = Gemma2Model.from_pretrained(
            self.model_repo, subfolder="text_encoder", torch_dtype=self.weight_dtype,
        ).to(device)
        vae = AutoencoderDC.from_pretrained(self.model_repo, subfolder="vae").to(device)
        transformer = SanaTransformer2DModel.from_pretrained(
            self.model_repo, subfolder="transformer", torch_dtype=self.weight_dtype,
        ).to(device)
        scheduler = SCMScheduler.from_pretrained(self.model_repo, subfolder="scheduler")

        pipe = SanaSprintPipeline(
            tokenizer=tokenizer, text_encoder=text_encoder,
            vae=vae, transformer=transformer, scheduler=scheduler,
        )

        self.model = pipe.transformer
        if self.use_grad_checkpoint:
            self.model.enable_gradient_checkpointing()

        self.vae = pipe.vae.to(torch.float32)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(
            num_inference_steps=self.sample_steps,
            max_timesteps=self.max_timesteps,
            intermediate_timesteps=self.intermediate_timesteps,
            device=device,
        )

        self.sigma_data = float(self.scheduler.config.sigma_data)
        self.guidance_scale_embed = float(self.model.config.guidance_embeds_scale)

        latent_size = int(self.model.config.sample_size)
        latent_ch = int(self.model.config.in_channels)
        self.latent_shape = (latent_ch, latent_size, latent_size)

        # Kept alive only for prompt encoding; released in free_text_encoder.
        self._pipe = pipe

    def encode_prompts(self, prompts: list[str], device) -> list[tuple]:
        """Pre-encode prompts → list of ``(prompt_embeds, prompt_attention_mask)``."""
        if self._pipe is None:
            raise RuntimeError("encode_prompts called after free_text_encoder().")
        cache = []
        for prompt in prompts:
            with torch.no_grad():
                embs, mask = self._pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    device=device,
                    max_sequence_length=self.max_sequence_length,
                    complex_human_instruction=self.complex_human_instruction,
                )
            cache.append((embs, mask))
        return cache

    def free_text_encoder(self) -> None:
        if self._pipe is not None:
            self._pipe.text_encoder = None
            self._pipe.tokenizer = None
            self._pipe = None
        torch.cuda.empty_cache()

    def inject_lora(self, *, rank, alpha, dropout, target_modules) -> tuple[int, int]:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(rank),
            lora_alpha=float(alpha),
            init_lora_weights="gaussian",
            target_modules=list(target_modules),
            lora_dropout=float(dropout),
        )
        self.model = get_peft_model(self.model, lora_config)
        self._peft_model = self.model

        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.to(torch.float32)

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        return n_train, n_total

    def get_lora_state(self) -> dict:
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(self._peft_model)

    def load_lora_state(self, state: dict) -> None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self._peft_model, state)

    def trainable_parameters(self) -> Iterable:
        return [p for p in self.model.parameters() if p.requires_grad]

    def prepare(self, accelerator) -> None:
        # _peft_model stays the underlying module so adapter toggling in
        # reference_context keeps working after DDP-wrapping.
        self.model = accelerator.prepare(self.model)

    @contextmanager
    def reference_context(self):
        self._peft_model.disable_adapter_layers()
        try:
            yield
        finally:
            self._peft_model.enable_adapter_layers()

    def make_noise(self, seeds, device):
        noise = torch.stack([
            torch.randn(self.latent_shape, generator=torch.manual_seed(int(s)))
            for s in seeds
        ]).to(device=device, dtype=self.weight_dtype)
        return noise * self.sigma_data

    def sample(self, noise, caption_embs, emb_masks, *, cfg_scale=None,
               grad_checkpoint_sampling: bool = False):
        """Differentiable SCM multi-step sampling. ``noise`` is sigma_data-scaled;
        returns VAE-scale latents (``denoised / sigma_data``)."""
        cfg_scale = self.cfg_scale if cfg_scale is None else cfg_scale
        device = noise.device
        bsz = noise.shape[0]
        sigma_data = self.sigma_data
        tdtype = self.model.dtype if hasattr(self.model, "dtype") else self.weight_dtype

        prompt_embeds = caption_embs.repeat(bsz, 1, 1).to(dtype=tdtype)
        prompt_mask = emb_masks.repeat(bsz, 1)

        guidance = torch.full([bsz], cfg_scale, device=device, dtype=tdtype)
        guidance = guidance * self.guidance_scale_embed

        # Reset the stateful scheduler for a fresh rollout.
        self.scheduler.set_begin_index(0)
        self.scheduler._step_index = None
        timesteps = self.scheduler.timesteps[:-1]

        use_ckpt = grad_checkpoint_sampling and torch.is_grad_enabled()
        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

        latents = noise
        denoised = noise
        for t in timesteps:
            timestep = t.expand(bsz).to(device)
            latents_model_input = latents / sigma_data

            scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
            scm_t_x = scm_t.view(-1, 1, 1, 1)
            scale = torch.sqrt(scm_t_x ** 2 + (1 - scm_t_x) ** 2)
            latent_model_input = latents_model_input * scale

            def _call(lat_in):
                return self.model(
                    lat_in.to(dtype=tdtype),
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    guidance=guidance,
                    timestep=scm_t,
                    return_dict=False,
                )[0]

            if use_ckpt:
                model_out = checkpoint(_call, latent_model_input, use_reentrant=False)
            else:
                model_out = _call(latent_model_input)

            noise_pred = (
                (1 - 2 * scm_t_x) * latent_model_input
                + (1 - 2 * scm_t_x + 2 * scm_t_x ** 2) * model_out
            ) / scale
            noise_pred = noise_pred.float() * sigma_data

            latents, denoised = self.scheduler.step(
                noise_pred, timestep, latents, return_dict=False,
            )

        return denoised / sigma_data

    def decode_latents(self, latents, *, enable_grad: bool):
        if enable_grad:
            return self._differentiable_vae_decode(latents.float())
        with torch.no_grad():
            z = latents.to(self.vae_dtype) / self.vae.config.scaling_factor
            return self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]

    def _differentiable_vae_decode(self, latent):
        """VAE decode with manual per-up_block gradient checkpointing
        (AutoencoderDC lacks native support)."""
        from torch.utils.checkpoint import checkpoint as ckpt

        scaling_factor = self.vae.config.scaling_factor or 0.41407
        z = latent / scaling_factor
        decoder = self.vae.decoder

        if decoder.in_shortcut:
            x = z.repeat_interleave(
                decoder.in_shortcut_repeats, dim=1,
                output_size=z.shape[1] * decoder.in_shortcut_repeats,
            )
            h = decoder.conv_in(z) + x
        else:
            h = decoder.conv_in(z)

        for up_block in reversed(decoder.up_blocks):
            h = ckpt(up_block, h, use_reentrant=False)

        h = decoder.norm_out(h.movedim(1, -1)).movedim(-1, 1)
        h = decoder.conv_act(h)
        h = decoder.conv_out(h)
        return h
