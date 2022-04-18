from dataclasses import dataclass
from typing import List


@dataclass
class BlockConfig:
    num_groups: int
    num_self_attn_layers: int
    num_self_attn_heads: int
    num_latents: int
    channels: int


@dataclass
class HiPConfig:
    input_dim: int
    blocks: List[BlockConfig]

