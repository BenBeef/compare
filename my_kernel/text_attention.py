import torch
from torch import nn
from .text_norm import Qwen3VLTextRMSNorm
from .transformer_utils import Cache
from .utils import apply_rotary_pos_emb, sdpa_attention_forward
from .text_attn_flash import Attention

class Qwen3VLTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
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
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3VLTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        self._q_out = config.num_attention_heads * self.head_dim      # 2048
        self._kv_out = config.num_key_value_heads * self.head_dim    # 1024

        # flash_attention
        self.attn = Attention(head_dim=self.head_dim, scale=self.scaling)

    def merge_params(self):
        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        total_out = self._q_out + self._kv_out * 2
        self.qkv_proj = nn.Linear(self.config.hidden_size, total_out, bias=False, device=device, dtype=dtype)
        self.qkv_proj.weight.data = torch.cat([
            self.q_proj.weight.data,
            self.k_proj.weight.data,
            self.v_proj.weight.data,
        ], dim=0).contiguous()
        del self.q_proj, self.k_proj, self.v_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)  # [seq, q_out + 2*kv_out]
        q, k, v = qkv.split([self._q_out, self._kv_out, self._kv_out], dim=-1)

        q = self.q_norm(q.view(seq_len, -1, self.head_dim))
        k = self.k_norm(k.view(seq_len, -1, self.head_dim))
        v = v.view(seq_len, -1, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos.squeeze(0), sin.squeeze(0))

        attn_output = self.attn(q, k, v)              # [seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(seq_len, -1)  # [seq_len, num_heads * head_dim]
        attn_output = self.o_proj(attn_output)
        return attn_output
