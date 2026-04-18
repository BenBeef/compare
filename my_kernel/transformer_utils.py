from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.utils.generic import maybe_autocast
# configs
from transformers.configuration_utils import PreTrainedConfig
from transformers import GenerationConfig