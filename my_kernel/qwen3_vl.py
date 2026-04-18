import os
from torch import nn
import torch
import time
from typing import *

from .vl_model import Qwen3VLVisionModel, BaseModelOutputWithDeepstackFeatures
from .text_model import Qwen3VLTextModel
from .transformer_utils import Cache, DynamicCache
from .transformer_utils import GenerationConfig
from dataclasses import dataclass
from .text_context import set_context, reset_context
from .pld_utils import pld_lookup_drafts
from .lookahead_n_gram import NGramMgr
import itertools


LOOKAHEAD_3_GRAM: NGramMgr = NGramMgr()

class PhaseTimer:
    """Global timer for profiling visual / prefill / decode phases."""
    def __init__(self):
        self.enabled = False
        self.records: dict[str, list[float]] = {
            'visual': [], 'prefill': [], 'decode': [], 'prepare': []
        }

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def record(self, phase: str, ms: float):
        if self.enabled:
            self.records[phase].append(ms)

    def summary(self) -> str:
        lines = ["\n" + "=" * 50, "  Phase Timing Summary", "=" * 50]
        for phase in ('visual', 'prefill', 'decode', 'prepare'):
            ts = self.records[phase]
            if ts:
                avg = sum(ts) / len(ts)
                lines.append(f"  {phase:>8s}:  avg {avg:7.2f} ms  (n={len(ts)})")
        lines.append("=" * 50)
        return "\n".join(lines)


TIMER = PhaseTimer()


@dataclass
class Qwen3VLCausalLMOutputWithPast:
    """
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


@dataclass
class Qwen3VLModelOutputWithPast:
    """
    """

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


class Qwen3VLModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)

        self.rope_deltas = None  # cache rope_deltas here
    
    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor
    ):
        """
        input_ids: [seq], inputs_embeds: [seq, hidden]
        Returns bool masks: image_mask [seq, hidden], video_mask [seq, hidden]
        """
        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return special_image_mask, special_video_mask

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [seq], mm_token_type_ids: [seq], attention_mask: [seq]
        Returns: position_ids [3, seq], rope_delta scalar tensor
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        device = input_ids.device

        current_input_ids = input_ids
        input_token_type = mm_token_type_ids
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask.bool()]
            input_token_type = input_token_type[attention_mask.bool()]

        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            input_type_group.append((key, group[0][0], group[-1][0] + 1))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                llm_pos_ids_list.append(
                    self.get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device=device)
                )
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # [3, effective_seq]

        position_ids = torch.zeros(3, len(input_ids), dtype=input_ids.dtype, device=device)
        if attention_mask is not None:
            position_ids[:, attention_mask.bool()] = llm_positions
        else:
            position_ids = llm_positions

        rope_delta = llm_positions.max() + 1 - len(current_input_ids)
        return position_ids, rope_delta

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None
    ) -> tuple | Qwen3VLModelOutputWithPast:
        """
        All tensors unbatched: input_ids [seq], inputs_embeds [seq, hidden], etc.
        """
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)  # [seq] → [seq, hidden]

        image_mask = None
        if pixel_values is not None:
            _t = TIMER.enabled
            if _t:
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            image_outputs: BaseModelOutputWithDeepstackFeatures = self.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = image_outputs.pooler_output
            deepstack_image_embeds = image_outputs.deepstack_features
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if _t:
                torch.cuda.synchronize()
                TIMER.record('visual', (time.perf_counter() - _t0) * 1000)
        
            # def get_shape(x):
            #     return list(x.shape)
            # print('---'*5, f'input_ids/pixel_values/image_embeds = {get_shape(input_ids)}/{get_shape(pixel_values)}/{get_shape(image_embeds)}')

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None:
            visual_pos_masks = image_mask[..., 0]  # [seq] bool
            deepstack_visual_embeds = deepstack_image_embeds

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None
    ) -> tuple | BaseModelOutputWithDeepstackFeatures:
        r"""
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
            The tensors corresponding to the input images.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output: BaseModelOutputWithDeepstackFeatures = self.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds

        return vision_output



class Qwen3VLForConditionalGeneration(nn.Module):


    # generate 相关参数
    _auto_class = None
    _is_stateful: bool = False
    main_input_name: str = "input_ids"
    device = torch.device('cuda:0')

    def __init__(self, config, block_size=256):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        self.generation_config = GenerationConfig.from_model_config(config)

        self.block_size = block_size
        self._vocab_size = config.text_config.vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
        """
        All tensors unbatched: input_ids [seq], hidden_states [seq, hidden], logits [seq, vocab].
        """
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids
        )

        hidden_states = outputs.last_hidden_state  # [seq, hidden]
        logits = self.lm_head(hidden_states)        # [seq, vocab]

        return Qwen3VLCausalLMOutputWithPast(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
    
    def set_prefill_context(self, seq_len:int):
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(list(range(seq_len)), dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_seqlen_q = max_seqlen_k = seq_len
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, None)
    
    def compute_block_tables(self, seq_len:int):
        block_tables = list(range(((seq_len + self.block_size) // self.block_size)))
        block_tables = torch.tensor([block_tables], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)  # [1, num_blocks]
        return block_tables

    def set_decode_context(self, seq_len:int):
        slot_mapping = torch.tensor([seq_len-1], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor([seq_len], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.compute_block_tables(seq_len)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

    # ---- CUDA Graph for decode ----

    def build_decode_graph(self):
        """Build CUDA graph for decode phase. Call once after model+KV cache init."""
        device = self.device
        max_blocks = 128

        # Static input buffers (written via fill_/copy_ before each replay)
        self._g_input_ids = torch.zeros(1, dtype=torch.long, device=device)
        self._g_position_ids = torch.zeros(4, 1, dtype=torch.long, device=device)

        # Static context buffers (shared with get_context() via set_context)
        self._g_slot_mapping = torch.zeros(1, dtype=torch.int32, device=device)
        self._g_context_lens = torch.ones(1, dtype=torch.int32, device=device)
        self._g_block_tables = torch.arange(max_blocks, dtype=torch.int32, device=device).unsqueeze(0)  # [1, max_blocks]

        # Point context to static buffers
        set_context(False,
                    slot_mapping=self._g_slot_mapping,
                    context_lens=self._g_context_lens,
                    block_tables=self._g_block_tables)

        # Warmup on side stream to prime allocator & JIT
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_forward()
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._decode_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph):
            self._g_next_token = self._decode_forward()

        # 3-token PLD graph only if enabled (saves capture + memory)
        if os.environ.get("USE_PLD", "0") == "1":
            self._build_decode_graph3()

    def _point_context_decode_g1(self):
        set_context(
            False,
            slot_mapping=self._g_slot_mapping,
            context_lens=self._g_context_lens,
            block_tables=self._g_block_tables,
        )

    def _point_context_decode_g3(self):
        set_context(
            False,
            slot_mapping=self._g3_slot_mapping,
            context_lens=self._g_context_lens,
            block_tables=self._g_block_tables,
        )

    def _build_decode_graph3(self):
        """CUDA graph for one forward over 3 new tokens (PLD verification)."""
        device = self.device
        vs = self._vocab_size
        self._g3_input_ids = torch.zeros(3, dtype=torch.long, device=device)
        self._g3_position_ids = torch.zeros(4, 3, dtype=torch.long, device=device)
        self._g3_slot_mapping = torch.zeros(3, dtype=torch.int32, device=device)
        self._g3_logits = torch.empty(3, vs, device=device, dtype=torch.float16)

        self._point_context_decode_g3()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_forward3()
        torch.cuda.current_stream().wait_stream(s)

        self._decode_graph3 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph3):
            self._g3_next_logits_argmax = self._decode_forward3()
        self._point_context_decode_g1()
    
    
    def _point_context_decode_g2(self):
        set_context(
            False,
            slot_mapping=self._g2_slot_mapping,
            context_lens=self._g_context_lens,
            block_tables=self._g_block_tables,
        )

    def _decode_forward2(self):
        """Three-token forward; copies logits for CPU verification after replay."""
        outputs = self.forward(
            input_ids=self._g2_input_ids,
            position_ids=self._g2_position_ids,
        )
        self._g2_logits.copy_(outputs.logits)
        return outputs.logits[0].argmax()
    
    def _build_decode_graph2(self):
        """CUDA graph for one forward over x new tokens (PLD verification)."""
        device = self.device
        vs = self._vocab_size
        self._g2_input_ids = torch.zeros(2, dtype=torch.long, device=device)
        self._g2_position_ids = torch.zeros(4, 2, dtype=torch.long, device=device)
        self._g2_slot_mapping = torch.zeros(2, dtype=torch.int32, device=device)
        self._g2_logits = torch.empty(2, vs, device=device, dtype=torch.float16)

        self._point_context_decode_g2()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_forward2()
        torch.cuda.current_stream().wait_stream(s)

        self._decode_graph2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph2):
            self._g2_next_logits_argmax = self._decode_forward2()
        
        # 默认
        self._point_context_decode_g1()

    def _point_context_decode_g4(self):
        set_context(
            False,
            slot_mapping=self._g4_slot_mapping,
            context_lens=self._g_context_lens,
            block_tables=self._g_block_tables,
        )

    def _decode_forward4(self):
        """Three-token forward; copies logits for CPU verification after replay."""
        outputs = self.forward(
            input_ids=self._g4_input_ids,
            position_ids=self._g4_position_ids,
        )
        self._g4_logits.copy_(outputs.logits)
        return outputs.logits[2].argmax()
    
    def _build_decode_graph4(self):
        """CUDA graph for one forward over x new tokens (PLD verification)."""
        device = self.device
        vs = self._vocab_size
        self._g4_input_ids = torch.zeros(4, dtype=torch.long, device=device)
        self._g4_position_ids = torch.zeros(4, 4, dtype=torch.long, device=device)
        self._g4_slot_mapping = torch.zeros(4, dtype=torch.int32, device=device)
        self._g4_logits = torch.empty(4, vs, device=device, dtype=torch.float16)

        self._point_context_decode_g4()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._decode_forward4()
        torch.cuda.current_stream().wait_stream(s)

        self._decode_graph4 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._decode_graph4):
            self._g4_next_logits_argmax = self._decode_forward4()
        
        # 默认
        self._point_context_decode_g1()

    def _decode_forward(self):
        """Single decode step on static graph buffers. Returns next_token (scalar)."""
        outputs = self.forward(
            input_ids=self._g_input_ids,
            position_ids=self._g_position_ids,
        )
        return outputs.logits[0].argmax()

    def _decode_forward3(self):
        """Three-token forward; copies logits for CPU verification after replay."""
        outputs = self.forward(
            input_ids=self._g3_input_ids,
            position_ids=self._g3_position_ids,
        )
        self._g3_logits.copy_(outputs.logits)
        return outputs.logits[2].argmax()

    def _accept_token_ids(self, words:List[int], preds:List[int], eos_ids):
        if preds[0] in eos_ids:
            return [], True
        
        # d1-> p1 总是有效decode
        accept_ids = [preds[0]]
        for i in range(len(preds)-1):
            pred = preds[i]
            word = words[i]
            next_token = preds[i+1]
            if pred != word:
                return accept_ids, False
            if pred in eos_ids:
                return accept_ids, True
            accept_ids.append(next_token)
        return accept_ids, False

    def _lookahead_decode_3(self, last_generated_token: int, kv_len, rope_delta_val, eos_ids, words:List[int]):
        if not hasattr(self, "_decode_graph3"):
                    self._build_decode_graph3()
        ip_0, ip_1, ip_2 = last_generated_token, words[0], words[1]
        self._g3_input_ids[0].fill_(ip_0)
        self._g3_input_ids[1].fill_(ip_1)
        self._g3_input_ids[2].fill_(ip_2)
        for c in range(3):
            p = kv_len + c + rope_delta_val
            self._g3_position_ids[:, c].fill_(p)
        self._g3_slot_mapping[0] = kv_len
        self._g3_slot_mapping[1] = kv_len + 1
        self._g3_slot_mapping[2] = kv_len + 2
        self._g_context_lens.fill_(kv_len + 3)

        self._point_context_decode_g3()
        self._decode_graph3.replay()
        torch.cuda.synchronize()


        preds = self._g3_logits.argmax(dim=-1).tolist()
        return self._accept_token_ids(words, preds, eos_ids)
    

    def _lookahead_decode_4(self, last_generated_token: int, kv_len, rope_delta_val, eos_ids, words:List[int]):
        if not hasattr(self, "_decode_graph4"):
                    self._build_decode_graph4()
        ip_0, ip_1, ip_2, ip_3 = last_generated_token, words[0], words[1], words[2]
        self._g4_input_ids[0].fill_(ip_0)
        self._g4_input_ids[1].fill_(ip_1)
        self._g4_input_ids[2].fill_(ip_2)
        self._g4_input_ids[3].fill_(ip_3)
        for c in range(4):
            p = kv_len + c + rope_delta_val
            self._g4_position_ids[:, c].fill_(p)
        self._g4_slot_mapping[0] = kv_len
        self._g4_slot_mapping[1] = kv_len + 1
        self._g4_slot_mapping[2] = kv_len + 2
        self._g4_slot_mapping[3] = kv_len + 3
        self._g_context_lens.fill_(kv_len + 4)

        self._point_context_decode_g4()
        self._decode_graph4.replay()
        torch.cuda.synchronize()


        preds = self._g4_logits.argmax(dim=-1).tolist()
        return self._accept_token_ids(words, preds, eos_ids)


    def _lookahead_decode_2(self, last_generated_token: int, kv_len, rope_delta_val, eos_ids, words:List[int]):
        if not hasattr(self, "_decode_graph2"):
                    self._build_decode_graph2()
        ip_0, ip_1 = last_generated_token, words[0]
        self._g2_input_ids[0].fill_(ip_0)
        self._g2_input_ids[1].fill_(ip_1)
        for c in range(2):
            p = kv_len + c + rope_delta_val
            self._g2_position_ids[:, c].fill_(p)
        self._g2_slot_mapping[0] = kv_len
        self._g2_slot_mapping[1] = kv_len + 1
        self._g_context_lens.fill_(kv_len + 2)

        self._point_context_decode_g2()
        self._decode_graph2.replay()
        torch.cuda.synchronize()


        preds = self._g2_logits.argmax(dim=-1).tolist()
        return self._accept_token_ids(words, preds, eos_ids)

    # ---- generate ----

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> torch.LongTensor:
        _t = TIMER.enabled
        if _t:
            torch.cuda.synchronize()
            _t_prep = time.perf_counter()

        input_ids = input_ids.squeeze(0)              # [seq]
        attention_mask = attention_mask.squeeze(0) if attention_mask is not None else None
        mm_token_type_ids = mm_token_type_ids.squeeze(0) if mm_token_type_ids is not None else None

        seq_len = input_ids.shape[0]
        device = input_ids.device

        eos_ids = self.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = {eos_ids}
        else:
            eos_ids = set(eos_ids) if eos_ids else set()

        # --- M-RoPE position_ids for prefill ---
        text_pos = torch.arange(seq_len, device=device)

        if mm_token_type_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            vision_pos, rope_delta = self.model.get_rope_index(
                input_ids, mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        else:
            vision_pos = text_pos.unsqueeze(0).expand(3, -1)
            rope_delta = torch.tensor(0, dtype=torch.long, device=device)

        self.model.rope_deltas = rope_delta
        position_ids = torch.cat([text_pos.unsqueeze(0), vision_pos], dim=0)  # [4, seq]

        if _t:
            torch.cuda.synchronize()
            TIMER.record('prepare', (time.perf_counter() - _t_prep) * 1000)

        # --- Build decode graph on first call (before any prefill) ---
        # use_pld = os.environ.get("USE_PLD", "0") == "1"
        use_pld = True
        if not hasattr(self, '_decode_graph'):
            self.build_decode_graph()
        if use_pld and not hasattr(self, '_decode_graph3'):
            self._build_decode_graph3()
        
        if use_pld and not hasattr(self, '_decode_graph2'):
            self._build_decode_graph2()
        if use_pld and not hasattr(self, '_decode_graph4'):
            self._build_decode_graph4()

        # --- Prefill (visual timing is inside Qwen3VLModel.forward) ---
        if _t:
            torch.cuda.synchronize()
            _t_prefill = time.perf_counter()

        self.set_prefill_context(seq_len)
        outputs = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        next_token_val = outputs.logits[-1, :].argmax(dim=-1).item()

        if _t:
            torch.cuda.synchronize()
            TIMER.record('prefill', (time.perf_counter() - _t_prefill) * 1000)

        generated_ids = [next_token_val]
        rope_delta_val = rope_delta.item()

        # --- Switch context to decode (point to static graph buffers) ---
        set_context(False,
                    slot_mapping=self._g_slot_mapping,
                    context_lens=self._g_context_lens,
                    block_tables=self._g_block_tables)

        # --- Decode loop with CUDA Graph replay ---
        if _t:
            torch.cuda.synchronize()
            _t_decode = time.perf_counter()

        input_ids_copy = input_ids.tolist()
        for i in range(len(input_ids_copy) -1, 1, -1):
            if input_ids_copy[i] > self.generation_config.eos_token_id:
                input_ids_copy = input_ids_copy[i+1:]
                break

        for _ in range(max_new_tokens - 1):
            
            # kv_len = seq_len  # tokens in KV before this step
            kv_len = seq_len + len(generated_ids) - 1 # tokens in KV before this step

            words = None

            if use_pld:
                last_generated_token = generated_ids[-1]
                words = LOOKAHEAD_3_GRAM.get_lookahead(last_generated_token)
                # if last_generated_token == 28715:
                    # words = [389, 279, 2383]
                    # print('------------')
            if (
                use_pld
                and words is not None
            ):
            # last_generated_token: int, kv_len, rope_delta_val, eos_ids, words
                accept_ids, stop = self._lookahead_decode_3(last_generated_token, kv_len, rope_delta_val, eos_ids, words)
                generated_ids.extend(accept_ids)
                if LOOKAHEAD_3_GRAM.collect:
                    LOOKAHEAD_3_GRAM.collect_hit(last_generated_token, len(words), len(accept_ids))
                if stop:
                    break
            
            # graph_1
            next_token_val = generated_ids[-1]
            all_seq_len = seq_len + len(generated_ids)

            self._point_context_decode_g1()
            if next_token_val in eos_ids:
                break

            # Fill static buffers with current step's values
            self._g_input_ids.fill_(next_token_val)
            self._g_position_ids.fill_(all_seq_len - 1 + rope_delta_val)
            self._g_slot_mapping.fill_(all_seq_len - 1)
            self._g_context_lens.fill_(all_seq_len)

            # Replay captured graph
            self._decode_graph.replay()

            # Read output (GPU→CPU sync for EOS check)
            next_token_val = self._g_next_token.item()
            generated_ids.append(next_token_val)

        if _t:
            torch.cuda.synchronize()
            TIMER.record('decode', (time.perf_counter() - _t_decode) * 1000)

        # 加入到lookahead词表
        LOOKAHEAD_3_GRAM.add_words(generated_ids)

        # Reassemble output
        gen_tensor = torch.tensor(generated_ids, dtype=torch.long, device=device)
        all_ids = torch.cat([input_ids, gen_tensor], dim=0)

        reset_context()
        return all_ids.unsqueeze(0)
    