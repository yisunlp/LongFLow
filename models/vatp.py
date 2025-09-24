from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb, eager_attention_forward
from transformers.cache_utils import Cache


logger = logging.get_logger(__name__)



class VATPAttention(nn.Module):
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
        attention_interface = eager_attention_forward
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
                key_states, value_states = past_key_value.initialize(key_states, value_states, attn_weights, self.layer_idx, attention_mask, cache_kwargs)
        else:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states, mask = past_key_value.update_stage1(key_states, value_states, self.layer_idx, cache_kwargs)
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
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update_stage2(key_states, value_states, attn_weights, self.layer_idx, cache_kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None

class VATPCache(Cache):
    def __init__(self, hh_size, recent_size) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.hh_score: List[torch.Tensor] = []
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.valid_mask = []

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
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_score: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        m1 = attn_score.sum(dim=2).mean(dim=1)
        m2 = value_states.norm(dim=-1).mean(dim=1)
        hh_score = m1 * m2
        bs, nh, seq_len, head_dim = key_states.shape
        if attention_mask is not None:
            attention_mask = attention_mask[:, 0, -1, :] == 0
        if seq_len > self.hh_size + self.recent_size:
            selected_hh_score = hh_score[:, :-self.recent_size]     
            _, topk = torch.topk(selected_hh_score.unsqueeze(1).repeat(1,nh,1), self.hh_size, dim=-1, largest=True)
            keep_idx = torch.cat([topk, torch.arange(seq_len-self.recent_size, seq_len,dtype=topk.dtype,device=topk.device).unsqueeze(0).unsqueeze(0).repeat(bs,nh,1)], dim=-1)
            key_states = torch.gather(key_states, dim=2, index=keep_idx.unsqueeze(-1).repeat(1, 1, 1, head_dim))
            value_states = torch.gather(value_states, dim=2, index=keep_idx.unsqueeze(-1).repeat(1, 1, 1, head_dim))
            hh_score = torch.gather(hh_score, dim=1, index=keep_idx[:,0,:])
            attention_mask = torch.gather(attention_mask, dim=1, index=keep_idx[:, 0, :])
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        self.hh_score.append(hh_score)
        self.valid_mask.append(attention_mask)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return None

    def update_stage1(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        self.valid_mask[layer_idx] = torch.cat([self.valid_mask[layer_idx], torch.ones([key_states.shape[0], key_states.shape[2]], dtype=self.valid_mask[layer_idx].dtype, device=self.valid_mask[layer_idx].device)], dim=-1)
        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.valid_mask[layer_idx]

    def update_stage2(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attn_score: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        m1 = attn_score.sum(dim=2).mean(dim=1)
        m2 = value_states.norm(dim=-1).mean(dim=1)
        hh_score = m1 * m2
        bs, nh, _, head_dim = key_states.shape
        self.hh_score[layer_idx] = hh_score+torch.cat([self.hh_score[layer_idx], torch.zeros([bs, 1],dtype=hh_score.dtype,device=hh_score.device)], dim=-1)
        kv_length = self.key_cache[layer_idx].shape[2]
        if kv_length > self.hh_size + self.recent_size:
            selected_hh_score = hh_score[:, :-self.recent_size]     
            _, topk = torch.topk(selected_hh_score.unsqueeze(1).repeat(1,nh,1), self.hh_size, dim=-1, largest=True)
            keep_idx = torch.cat([topk, torch.arange(kv_length-self.recent_size, kv_length,dtype=topk.dtype,device=topk.device).unsqueeze(0).unsqueeze(0).repeat(bs,nh,1)], dim=-1)
            self.key_cache[layer_idx] = torch.gather(self.key_cache[layer_idx], dim=2, index=keep_idx.unsqueeze(-1).repeat(1, 1, 1, head_dim))
            self.value_cache[layer_idx] = torch.gather(self.value_cache[layer_idx], dim=2, index=keep_idx.unsqueeze(-1).repeat(1, 1, 1, head_dim))
            self.hh_score[layer_idx] = torch.gather(self.hh_score[layer_idx], dim=1, index=keep_idx[:,0,:])
            self.valid_mask[layer_idx] = torch.gather(self.valid_mask[layer_idx], dim=1, index=keep_idx[:,0,:])
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

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

