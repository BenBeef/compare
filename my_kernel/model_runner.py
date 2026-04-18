import torch
from torch import nn
from .configs import Qwen3VLConfig

class ModelRunner:

    def __init__(self, model: nn.Module, config: Qwen3VLConfig, dtype, device, block_size=256) -> None:
        
        self.model = model
        # text kv_cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.block_size = block_size
        num_hidden_layers = config.text_config.num_hidden_layers
        num_kv_heads = config.text_config.num_key_value_heads
        num_kvcache_blocks = 512
        head_dim = config.text_config.head_dim
        self.kv_cache = torch.empty(2, num_hidden_layers, num_kvcache_blocks, self.block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        torch.cuda.empty_cache()