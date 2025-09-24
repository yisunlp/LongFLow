import transformers
from .sink import SinkAttention, SinkCache
from .h2o import H2OAttention, H2OCache
from .ours import OursAttention, OursCache
from .rkv import RKVAttention, RKVCache
from .vatp import VATPAttention, VATPCache

def replace_sink():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = SinkAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = SinkAttention.forward
    return SinkCache

def replace_h2o():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = H2OAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = H2OAttention.forward
    return H2OCache

def replace_ours():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = OursAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = OursAttention.forward
    return OursCache

def replace_rkv():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = RKVAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = RKVAttention.forward
    return RKVCache

def replace_vatp():
    transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward = VATPAttention.forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = VATPAttention.forward
    return VATPCache