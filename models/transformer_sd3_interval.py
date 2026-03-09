# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import WEIGHTS_NAME 
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
import safetensors
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
import math

from .attention import JointTransformerBlock

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class LogitNormalCosineScheduler:
    """
    Combined Logit-Normal timestep sampling + Cosine interpolation scheduling.
    This is the optimal approach used in Stable Diffusion 3.
    """
    
    def __init__(self, loc: float = 0.0, scale: float = 1.0, min_t: float = 1e-4, max_t: float = 1.0 - 1e-4):
        """
        Args:
            loc: Location parameter (mu) for logit-normal - 0.0 for symmetric around t=0.5
            scale: Scale parameter (sigma) for logit-normal - 1.0 is SD3 default
            min_t: Minimum timestep to avoid singularities
            max_t: Maximum timestep to avoid singularities
        """
        self.loc = loc
        self.scale = scale
        self.min_t = min_t
        self.max_t = max_t
        # Create LogitNormal using TransformedDistribution
        base_normal = Normal(loc, scale)
        self.logit_normal = TransformedDistribution(base_normal, SigmoidTransform())
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timesteps from logit-normal distribution."""
        # Step 1: Sample from logit-normal distribution
        t = self.logit_normal.sample((batch_size,)).to(device)
        
        # Step 2: Clamp to avoid singularities at 0 and 1
        t = torch.clamp(t, self.min_t, self.max_t)
        
        return t
    
    def get_cosine_schedule_params(self, t: torch.Tensor, sigma_min: float = 1e-6) -> tuple:
        """
        Convert logit-normal sampled timesteps to cosine-scheduled interpolation parameters.
        
        Args:
            t: Logit-normal sampled timesteps [batch_size]
            sigma_min: Minimum noise level
            
        Returns:
            alpha_t, sigma_t: Cosine-scheduled interpolation parameters
        """
        # Apply cosine scheduling transformation
        t_cos = 0.5 * (1 - torch.cos(math.pi * t))
        
        # Cosine interpolation parameters
        alpha_t = t_cos
        # sigma_t = 1 - t_cos + sigma_min * t_cos
        sigma_t = 1 - t_cos * (1 - sigma_min)
        
        return alpha_t, sigma_t

    def get_velocity_target(self, x1: torch.Tensor, z: torch.Tensor, sigma_min: float = 1e-6) -> torch.Tensor:
        """
        Compute velocity target for cosine-scheduled flow matching.
        
        Args:
            x1: Clean data
            z: Noise  
            t: Logit-normal sampled timesteps
            sigma_min: Minimum noise level
            
        Returns:
            u: Velocity target
        """
        # For cosine scheduling, the velocity target is:
        u = x1 - (1 - sigma_min) * z
        return u

    def create_cosine_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Create cosine-scheduled timestep sequence for inference."""
        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        # Apply cosine transformation for smoother scheduling
        t_span = 0.5 * (1 - torch.cos(math.pi * t_span))
        return t_span

class CombinedTimestepTextProjEmbeddings_interval(nn.Module):
    def __init__(self, embedding_dim, pooled_projection_dim, use_interval_mlp: bool = True):
        super().__init__()

        self.use_interval_mlp = use_interval_mlp
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=embedding_dim, pooled_projection_dim=pooled_projection_dim
        )
        if use_interval_mlp:
            self.interval_embedder = nn.Sequential(
                nn.Linear(2 * embedding_dim, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        else:
            self.interval_embedder = None  # r_emb + t_emb

    def forward(self, r_timestep, t_timestep, pooled_projection):
        # r
        temb_r = self.time_text_embed(r_timestep, pooled_projection)
        temb_t = self.time_text_embed(t_timestep, pooled_projection)
        combined = torch.cat([temb_r, temb_t], dim=-1)

        temb = self.interval_embedder(combined)

        return temb
    
class SD3Transformer2DModel_interval(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        # self.time_text_embed = CombinedTimestepTextProjEmbeddings(
        #     embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        # )
        # self.time_text_embed_t = CombinedTimestepTextProjEmbeddings(
        #     embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        # )
        self.time_text_embed_interval = CombinedTimestepTextProjEmbeddings_interval(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.interval_embedder = nn.Sequential(
            nn.Linear(2 * self.inner_dim, self.inner_dim),  # [r, t] -> dim
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim)
        )

    # # Initialize interval embedder:
    def initialize_weights(self):
        # Initialize to near-zero for stability
        nn.init.normal_(self.interval_embedder[0].weight, std=0.02)
        # nn.init.zeros_(self.interval_embedder[0].bias)
        nn.init.normal_(self.interval_embedder[2].weight, std=0.02)
        # nn.init.zeros_(self.interval_embedder[2].bias)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedJointAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_image: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        r_timestep: torch.LongTensor = None,
        timestep: torch.LongTensor = None, # t_timestep
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        # temb_r = self.time_text_embed(r_timestep, pooled_projections)
        # temb_t = self.time_text_embed(timestep, pooled_projections)
        # combined = torch.cat([temb_r, temb_t], dim=-1)
        # temb = self.interval_embedder(combined)
        # temb = temb_r + temb_t
        temb = self.time_text_embed_interval(r_timestep, timestep, pooled_projections)
        # temb_t = self.time_text_embed(timestep, pooled_projections)
        # Interval embedding: encode [r, t] information
        # combined = torch.cat([temb_r, temb_t], dim=-1)
        # interval = torch.stack([r_timestep, timestep], dim=-1)  # (N, 2)
        # interval_emb = self.interval_embedder(interval)  # (N, D)
        # temb = temb + interval_emb

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ## control_dit begin
        controlnet_image = self.pos_embed(controlnet_image)
        hidden_states = torch.cat([hidden_states, controlnet_image], dim=-2)
        ## control_dit end
        
        for index_block, block in enumerate(self.transformer_blocks):
            # Skip specified layers
            is_skip = True if skip_layers is not None and index_block in skip_layers else False

            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb
                )


            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        # control_dit begin 
        hidden_states, hidden_states_control = hidden_states.chunk(2, dim=1)
        # control_dit end

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained_local(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        model = cls.from_config(config)
        
        model_file = os.path.join(pretrained_model_path, SAFETENSORS_WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = safetensors.torch.load_file(model_file, device="cpu")
        model.load_state_dict(state_dict, strict=False)


        print("Successfully loaded Transformer weights!")
        return model