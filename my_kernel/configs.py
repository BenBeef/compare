from typing import TYPE_CHECKING, Optional, TypedDict
from .transformer_utils import PreTrainedConfig

class RopeParameters(TypedDict, total=False):
    """
    Args:
        rope_theta (`float`):
            The base period of the RoPE embeddings.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        partial_rotary_factor (`float`, *optional*):
            The percentage of the query and key head embedding on which RoPE will be applied.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'yarn', 'longrope' and 'llama3'. The original max position embeddings used during
            pretraining.
        attention_factor (`float`, *optional*):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        short_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        long_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    rope_theta: float
    rope_type: str | None
    partial_rotary_factor: float | None
    factor: float | None
    original_max_position_embeddings: int | None
    attention_factor: float | None
    beta_fast: float | None
    beta_slow: float | None
    short_factor: list[float] | None
    long_factor: list[float] | None
    low_freq_factor: float | None
    high_freq_factor: float | None


class Qwen3VLVisionConfig(PreTrainedConfig):
    r"""
    out_hidden_size (`int`, *optional*, defaults to 3584):
        The output hidden size of the vision model.
    num_position_embeddings (`int`, *optional*, defaults to 2304):
        The maximum sequence length that this model might ever be used with
    deepstack_visual_indexes (`list[int]`, *optional*, defaults to `[8, 16, 24]`):
        Indexed of layers for deepstack embeddings.
    """

    model_type = "qwen3_vl"
    base_config_key = "vision_config"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] | tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02




class Qwen3VLTextConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen3VLTextModel, Qwen3VLTextConfig

    >>> # Initializing a Qwen3VL style configuration
    >>> configuration = Qwen3VLTextConfig()

    >>> # Initializing a model from the Qwen3-VL-7B style configuration
    >>> model = Qwen3VLTextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"
    default_theta = 500000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


class Qwen3VLConfig(PreTrainedConfig):
    r"""
    Example:

    ```python
    >>> from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig

    >>> # Initializing a Qwen3-VL style configuration
    >>> configuration = Qwen3VLConfig()

    >>> # Initializing a model from the Qwen3-VL-4B style configuration
    >>> model = Qwen3VLForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__post_init__(**kwargs)