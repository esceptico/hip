from torch import nn

from hip.block import PerceiverBlock
from hip.config import HiPConfig


class HiP(nn.Module):
    def __init__(self, config: HiPConfig):
        super().__init__()
        layers = []
        input_dim = config.input_dim
        for block in config.blocks:
            layer = PerceiverBlock(
                input_dim=input_dim,
                num_groups=block.num_groups,
                num_self_attn_layers=block.num_self_attn_layers,
                num_self_attn_heads=block.num_self_attn_heads,
                num_latents=block.num_latents,
                channels=block.channels
            )
            layers.append(layer)
            input_dim = block.channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x, attention_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
            attention_mask = None
        return x
