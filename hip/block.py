from typing import Optional

import torch
from torch import nn

from hip.attention import CrossAttention, SelfAttention


class PerceiverBlock(nn.Module):
    """Basic Hierarchical Perceiver block. Consists of learned set of latent vectors (one for each group),
    cross-attention encoding layer and number of self-attention processing layers.
    All parameters of cross- and self-attention layers are shared.
    """
    def __init__(
        self,
        input_dim: int,
        num_groups: int,
        num_latents: int,
        channels: int,
        num_self_attn_layers: int = 1,
        num_cross_attn_heads: int = 1,
        num_self_attn_heads: int = 1,
        qk_out_dim: Optional[int] = None,
        v_out_dim: Optional[int] = None,
        cross_attn_widening_factor: int = 1,
        self_attn_widening_factor: int = 1,
        use_query_residual: bool = True,
        dropout: float = 0.0,
        cross_attn_dropout: float = 0.0,
        self_attn_dropout: float = 0.0
    ):
        super().__init__()
        self.num_groups = num_groups

        self.latents = nn.Parameter(torch.randn(num_groups, num_latents, channels))
        self.cross_attention = CrossAttention(
            kv_dim=input_dim,
            q_dim=channels,
            num_heads=num_cross_attn_heads,
            dropout=dropout,
            attention_dropout=cross_attn_dropout,
            widening_factor=cross_attn_widening_factor,
            use_query_residual=use_query_residual,
            qk_out_dim=qk_out_dim,
            v_out_dim=v_out_dim
        )
        self.self_attention_layers = nn.ModuleList([
            SelfAttention(
                hidden_dim=channels,
                num_heads=num_self_attn_heads,
                dropout=dropout,
                attention_dropout=self_attn_dropout,
                widening_factor=self_attn_widening_factor,
                qk_out_dim=qk_out_dim,
                v_out_dim=v_out_dim
            ) for _ in range(num_self_attn_layers)
        ])

    def forward(self, inputs, attention_mask=None):
        *dims, seq_len, input_dim = inputs.size()
        if attention_mask is not None:
            # (bs, seq_len) -> (bs, num_groups, group_len)
            attention_mask = attention_mask.view(*dims, self.num_groups, -1)
            # (bs, num_groups, group_len) -> (bs, num_groups, num_heads, q_seq_len, kv_seq_len)
            # num_groups and q_seq_len are broadcast
            # group_len is the same as kv_seq_len
            attention_mask = attention_mask[:, :, None, None, :]

        # (..., seq_len, hid_dim) -> (..., num_groups, group_len, hid_dim)
        inputs = inputs.view(*dims, self.num_groups, -1, input_dim)
        latents = self.cross_attention(inputs, self.latents, attention_mask)
        for self_attention in self.self_attention_layers:
            latents = self_attention(latents)

        # (.., num_groups, group_len, latent_dim) -> (.., seq_len, hid_dim)
        *_, latents_dim = latents.size()
        outputs = latents.view(*dims, -1, latents_dim)
        return outputs
