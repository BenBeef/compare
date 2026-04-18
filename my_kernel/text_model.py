from torch import nn
import torch

import torch 
from torch import nn
from .text_layer import Qwen3VLTextDecoderLayer
from .text_norm import Qwen3VLTextRMSNorm
from .text_embed import Qwen3VLTextRotaryEmbedding
from .transformer_utils import Cache, create_causal_mask
from dataclasses import dataclass


@dataclass
class BaseModelOutputWithPast:
    """
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class Qwen3VLTextModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        # args for deepstack
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None
    ) -> tuple | BaseModelOutputWithPast:
        """
        All tensors are unbatched: input_ids [seq], inputs_embeds [seq, hidden], etc.
        Batch dim is added temporarily only where required (causal mask, RoPE).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)  # [seq] → [seq, hidden]

        if position_ids.ndim == 2 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]                     # [3, seq] for MRoPE

        hidden_states = inputs_embeds  # [seq, hidden]

        # RoPE: position_ids [3, seq] → [3, 1, seq] for rotary_emb
        position_embeddings = self.rotary_emb(hidden_states.unsqueeze(0), position_ids.unsqueeze(1))

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings
            )
            hidden_states = layer_outputs

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        # hidden_states: [seq, hidden], visual_pos_masks: [seq] bool
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + visual_embeds
        return hidden_states