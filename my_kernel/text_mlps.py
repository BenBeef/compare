import torch
from torch import nn
from .activations import SiluAndMul

class Qwen3VLTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # self.act_fn = ACT2FN[config.hidden_act]
        self.act_fn = SiluAndMul()
    
    def merge_params(self):
        device = self.gate_proj.weight.device
        dtype = self.gate_proj.weight.dtype
        self.gate_up_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False, device=device, dtype=dtype)
        self.gate_up_proj.weight.data = torch.cat([self.gate_proj.weight.data, self.up_proj.weight.data], dim=0).contiguous()
        del self.gate_proj, self.up_proj
    
    def forward(self, x):
        gate_up_x = self.gate_up_proj(x)
        gate_up_x = self.act_fn(gate_up_x)
        down_proj = self.down_proj(gate_up_x)
        return down_proj