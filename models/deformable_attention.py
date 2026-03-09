"""Lightweight multi-scale deformable attention module for RelTR DINOv3."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, xavier_uniform_


class MultiScaleDeformableAttention(nn.Module):
    """PyTorch implementation of multi-scale deformable attention.

    This is a CPU-friendly port inspired by the official Deformable DETR module.
    It supports a configurable number of feature levels and sampling points but
    defaults to a single level for compatibility with the current RelTR decoder.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int = 1, n_points: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.n_levels = n_levels
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply deformable attention.

        Args:
            query: [B, Len_q, C]
            reference_points: [B, Len_q, n_levels, 2]
            input_flatten: [B, \sum_l H_l W_l, C]
            input_spatial_shapes: [n_levels, 2]
            input_level_start_index: [n_levels]
            input_padding_mask: optional [B, \sum_l H_l W_l]
        """
        B, Len_q, C = query.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(B, -1, self.num_heads, C // self.num_heads)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, Len_q, self.num_heads, self.n_levels, self.n_points, 2)

        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Len_q, self.num_heads, self.n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, -1)
        attn_weights = attn_weights.view(B, Len_q, self.num_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] != 2:
            raise ValueError("reference_points last dimension must be 2")

        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], dim=-1
        )  # [L,2]
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = self._ms_deformable_attn_pytorch(
            value,
            input_spatial_shapes,
            sampling_locations,
            attn_weights,
            input_level_start_index,
        )
        output = output.view(B, Len_q, self.embed_dim)
        return self.output_proj(output)

    def _ms_deformable_attn_pytorch(
        self,
        value: torch.Tensor,
        spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attn_weights: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Pure PyTorch implementation following Deformable DETR."""
        B, Len_in, n_heads, head_dim = value.shape
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        value_list = value.split([H_ * W_ for H_, W_ in spatial_shapes.tolist()], dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []

        for lid, (H_l, W_l) in enumerate(spatial_shapes.tolist()):
            value_l = value_list[lid]
            value_l = value_l.view(B, H_l * W_l, n_heads, head_dim)
            value_l = value_l.permute(0, 2, 3, 1).view(B * n_heads, head_dim, H_l, W_l)

            sampling_grid_l = sampling_grids[:, :, :, lid].transpose(1, 2).reshape(B * n_heads, Len_q, n_points, 2)
            sampling_feat = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_feat = sampling_feat.reshape(B, n_heads, head_dim, Len_q, n_points)
            sampling_feat = sampling_feat.permute(0, 3, 1, 4, 2)
            sampling_value_list.append(sampling_feat)

        sampling_value = torch.stack(sampling_value_list, dim=3)
        output = (sampling_value * attn_weights.unsqueeze(-1)).sum(-2).sum(-2)
        output = output.reshape(B, Len_q, self.num_heads * head_dim)
        return output