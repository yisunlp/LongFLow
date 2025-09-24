from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, eager_attention_forward, repeat_kv
from transformers.cache_utils import Cache
import math


import torch.nn.functional as F
logger = logging.get_logger(__name__)



class RKVAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        if not (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        if query_states.shape[-2] != 1:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.initialize(query_states, key_states, value_states, attention_mask, self.layer_idx, cache_kwargs)
        else:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states, mask = past_key_value.update(query_states, key_states, value_states, self.layer_idx, cache_kwargs)
                mask = (~mask).unsqueeze(1).unsqueeze(2).to(key_states.dtype)
                mask = mask * torch.finfo(mask.dtype).min
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class RKVCache(Cache):
    def __init__(self, cache_budget) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.cache_budget = cache_budget
        self.buffer_size = 128
        self.alpha = 8
        self._lambda = 0.1
        self.hh_budget = int(self.cache_budget - self.buffer_size - self.alpha / 2)
        self.query_cache: List[torch.Tensor] = []
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.count = 0

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def initialize(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        bs, num_heads, seq_len, head_dim = key_states.shape
        if attention_mask is None:
            mask = torch.ones((bs, seq_len), device=key_states.device, dtype=torch.bool)
        else:
            mask = attention_mask[:,0,-1,:]==0
        self.query_cache.append(query_states[:,:,-self.alpha:,:])
        if key_states.shape[-2] > self.hh_budget + self.buffer_size:
            window_query = self.query_cache[layer_idx]
            key_to_drop = key_states[:,:,:-self.buffer_size,:]
            value_to_drop = value_states[:,:,:-self.buffer_size,:]
            mask_to_drop = mask[:,:-self.buffer_size]
            key_to_drop, value_to_drop, mask_to_drop = self.evict(window_query, key_to_drop, value_to_drop, self.hh_budget - self.buffer_size, mask_to_drop)
            key_to_drop = torch.cat([key_to_drop, key_states[:,:,-self.buffer_size:,:]], dim=-2)
            value_to_drop = torch.cat([value_to_drop, value_states[:,:,-self.buffer_size:,:]], dim=-2)
            mask_to_drop = torch.cat([mask_to_drop, mask[:,-self.buffer_size:]], dim=-1)
        else:
            key_to_drop = key_states
            value_to_drop = value_states
            mask_to_drop = mask
        self.key_cache.append(key_to_drop)
        self.value_cache.append(value_to_drop)
        self.masks.append(mask_to_drop)
        return key_states, value_states

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `StaticCompressCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx][:,:,-self.alpha:,:], query_states[:,:,-1:,:]], dim=-2)
        key_states = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        value_states = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        mask = torch.cat([self.masks[layer_idx], torch.ones((key_states.shape[0], 1), device=key_states.device, dtype=torch.bool)], dim=-1)
        if key_states.shape[-2] > self.hh_budget + self.buffer_size:
            window_query = self.query_cache[layer_idx]
            key_to_drop = key_states[:,:,:-self.buffer_size,:]
            value_to_drop = value_states[:,:,:-self.buffer_size,:]
            mask_to_drop = mask[:,:-self.buffer_size]
            key_to_drop, value_to_drop, mask_to_drop = self.evict(window_query, key_to_drop, value_to_drop, self.hh_budget - self.buffer_size, mask_to_drop)
            key_to_drop = torch.cat([key_to_drop, key_states[:,:,-self.buffer_size:,:]], dim=-2)
            value_to_drop = torch.cat([value_to_drop, value_states[:,:,-self.buffer_size:,:]], dim=-2)
            mask_to_drop = torch.cat([mask_to_drop, mask[:,-self.buffer_size:]], dim=-1)
        else:
            key_to_drop = key_states
            value_to_drop = value_states
            mask_to_drop = mask
        self.key_cache[layer_idx] = key_to_drop
        self.value_cache[layer_idx] = value_to_drop
        self.masks[layer_idx] = mask_to_drop
        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.masks[layer_idx]


    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or not self.key_cache[layer_idx].numel()  # the layer has no cache
        )
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def evict(self, query, key, value, budget, mask):
        head_dim=key.shape[-1]
        num_groups=query.shape[1]//key.shape[1]
        key_repeat=repeat_kv(key,num_groups)
        attn_weights = torch.matmul(query, key_repeat.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = (~mask).unsqueeze(1).unsqueeze(2).to(query.dtype)
        attention_mask = attention_mask * torch.finfo(attention_mask.dtype).min
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = attn_weights.sum(dim=-2)
        I_score = F.max_pool1d(attn_weights, kernel_size = 5, padding=2, stride=1)

        normed_key = key / key.square().sum(dim=-1, keepdim=True)
        sim = torch.matmul(normed_key, normed_key.transpose(2, 3))
        R_score = torch.softmax(sim.mean(dim=-1),dim=-1)
        
        score = self._lambda * I_score.mean(dim=1) - (1 - self._lambda) * R_score.mean(dim=1)
        indices = torch.topk(score, k=budget, dim=-1).indices
        mask = torch.gather(mask, dim=1, index=indices)
        indices = indices.unsqueeze(1).unsqueeze(-1).expand([-1, key.shape[1], -1, head_dim])
        key=torch.gather(key, dim=2, index=indices)
        value=torch.gather(value, dim=2, index=indices)
        return key,value,mask
