from collections import OrderedDict
import torch
from torch import nn
from .graph_data import image_grid_thw_rate
from tqdm import tqdm

CNT_CACHE_HIT = False
HIT = 0
TOTAL = 0

BUILD_CACHE = True

PIX_VAL_SHAPE_1 = 1536
MAX_VG_CNT = 16

class VisualCudaGraph:

    def __init__(self, key, visual_model: nn.Module) -> None:
        self.key = key
        self.visual_model = visual_model
        self._graph: torch.cuda.CUDAGraph | None = None

        self.pos_embeds: torch.Tensor = None
        self.rotary_pos_emb: torch.Tensor = None
        self.cu_seqlens: torch.Tensor = None

        self._buf_input: torch.Tensor = None
        self._out_pooler: torch.Tensor = None
        self._out_hidden: torch.Tensor = None
        self._out_deepstack: list[torch.Tensor] = None

    def _prepare_precomputed(self, grid_thw: torch.Tensor):
        self.pos_embeds = self.visual_model.fast_pos_embed_interpolate(grid_thw)
        self.rotary_pos_emb = self.visual_model.rot_pos_emb(grid_thw)
        self.cu_seqlens = self.visual_model.get_cu_seqlens(grid_thw)

    def _capture(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        self._prepare_precomputed(grid_thw)

        if BUILD_CACHE:
            self._buf_input = pixel_values.clone()

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.visual_model(self._buf_input, grid_thw=grid_thw, vg=self)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self.visual_model(self._buf_input, grid_thw=grid_thw, vg=self)
            self._graph = g

        else:
            out = self.visual_model(pixel_values, grid_thw=grid_thw, vg=self)

        self._out_hidden = out.last_hidden_state
        self._out_pooler = out.pooler_output
        self._out_deepstack = out.deepstack_features

    def run(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        from .vl_model import BaseModelOutputWithDeepstackFeatures
        global TOTAL, HIT

        if CNT_CACHE_HIT:
            TOTAL += 1

        if self._graph is None:
            self._capture(pixel_values, grid_thw)
        else:
            if CNT_CACHE_HIT:
                HIT += 1
            self._buf_input.copy_(pixel_values)
            self._graph.replay()

        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=self._out_hidden,
            pooler_output=self._out_pooler,
            deepstack_features=self._out_deepstack,
        )


class VisualGraphMgr:

    def __init__(self, capacity: int = MAX_VG_CNT) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[tuple, VisualCudaGraph] = OrderedDict()

    def get(self, visual_model: nn.Module, grid_thw: torch.Tensor) -> VisualCudaGraph:
        key = tuple(grid_thw.flatten().tolist())

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        vg = VisualCudaGraph(key, visual_model)
        self._cache[key] = vg

        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

        return vg
    
    def show_cache_hit(self, clear=True):
        global HIT, TOTAL
        print(f'visual graph cache hit: {HIT} / {TOTAL} = {HIT/TOTAL if TOTAL else None}')
        if clear:
            HIT = 0
            TOTAL = 0
    
    def run_warmup(self, visual_model: nn.Module, device):
        global BUILD_CACHE
        BUILD_CACHE = True

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for img_grid, _, _  in tqdm(image_grid_thw_rate[:MAX_VG_CNT], desc="visual_model_warmup"):
            pixel_values = torch.randn(img_grid[0]*img_grid[1]*img_grid[2], PIX_VAL_SHAPE_1, dtype=torch.float32, device=device)
            img_grid = torch.tensor([img_grid], dtype=torch.int64, device=device)
            vg = self.get(visual_model, img_grid)
            vg.run(pixel_values, img_grid)

        torch.cuda.empty_cache()
        BUILD_CACHE = False

VG_MGR = VisualGraphMgr()