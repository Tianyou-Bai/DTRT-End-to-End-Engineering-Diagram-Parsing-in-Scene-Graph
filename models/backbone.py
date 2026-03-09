# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from .dynamic_swin import AdaSwinTransformer

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class SwinBackbone(nn.Module):
    """Swin Transformer backbone."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()
        # Load Swin Transformer from torchvision
        # Note: dilation is not directly supported in standard Swin implementation
        if name == 'swin_t':
            weights = torchvision.models.Swin_T_Weights.DEFAULT if is_main_process() else None
            backbone = torchvision.models.swin_t(weights=weights)
            embed_dim = 96
        elif name == 'swin_s':
            weights = torchvision.models.Swin_S_Weights.DEFAULT if is_main_process() else None
            backbone = torchvision.models.swin_s(weights=weights)
            embed_dim = 96
        elif name == 'swin_b':
            weights = torchvision.models.Swin_B_Weights.DEFAULT if is_main_process() else None
            backbone = torchvision.models.swin_b(weights=weights)
            embed_dim = 128
        else:
            raise ValueError(f"Unsupported swin backbone: {name}")

        # Swin-T: 
        # features.0 : PatchEmbed
        # features.1 : Stage 1
        # features.2 : PatchMerging
        # features.3 : Stage 2
        # features.4 : PatchMerging
        # features.5 : Stage 3
        # features.6 : PatchMerging
        # features.7 : Stage 4
        
        # We need output from stages. Keys in return_layers must match module names.
        # IntermediateLayerGetter iterate over named_children(). 
        # For Swin, we should wrap backbone.features to access "1", "3", etc.
        if return_interm_layers:
            return_layers = {"1": "0", "3": "1", "5": "2", "7": "3"}
            self.strides = [4, 8, 16, 32]
            self.num_channels = embed_dim * 8 # Last feature map channels
        else:
            return_layers = {"7": "0"}
            self.strides = [32]
            self.num_channels = embed_dim * 8 # 768 for Swin-T
            
        self.body = IntermediateLayerGetter(backbone.features, return_layers=return_layers)
        self.norm = backbone.norm
        
        # Freeze params if needed (Not implemented for Swin specifically in this template, user usually fine-tunes all)
        if not train_backbone:
             for name, parameter in self.body.named_parameters():
                 parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            # Apply final LayerNorm if this is the last stage
            if name == "3" or (len(xs) == 1 and name == "0"):
                 x = self.norm(x)
            
            # Swin outputs [B, H, W, C], convert to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
            
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class DynamicSwinBackbone(nn.Module):
    """Dynamic Swin Transformer backbone."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        super().__init__()
        
        if name == 'dynamic_swin_t':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'dynamic_swin_s':
            embed_dim = 96
            depths = [2, 2, 18, 2]
            num_heads = [3, 6, 12, 24]
        elif name == 'dynamic_swin_b':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
        else:
            raise ValueError(f"Unsupported dynamic swin backbone: {name}")

        self.body = AdaSwinTransformer(
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_path_rate=0.2 if name == 'dynamic_swin_b' else 0.1 # Adjust based on model size
        )
        
        if return_interm_layers:
            self.num_channels = embed_dim * 8
        else:
            self.num_channels = embed_dim * 8
            
        self.return_interm_layers = return_interm_layers
        
        if not train_backbone:
             for name, parameter in self.body.named_parameters():
                 parameter.requires_grad_(False)

    def forward(self, tensor_list: NestedTensor):
        # AdaSwinTransformer returns (outs, decisions, scores)
        # outs is a dict {"0": x1, "1": x2, ...} where x is [B, C, H, W]
        outs, decisions, scores = self.body(tensor_list.tensors)
        
        out: Dict[str, NestedTensor] = {}
        
        keys = ["0", "1", "2", "3"] if self.return_interm_layers else ["3"]
        
        for key in keys:
            if key in outs:
                x = outs[key]
                m = tensor_list.mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[key] = NestedTensor(x, mask)
                
        return out, decisions, scores

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        out_backbone = self[0](tensor_list)
        if isinstance(out_backbone, tuple) and len(out_backbone) == 3:
             xs, decisions, scores = out_backbone
        else:
             xs = out_backbone
             decisions, scores = None, None

        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        if decisions is not None:
            return out, pos, decisions, scores
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = getattr(args, "masks", False) or getattr(args, "return_interm_layers", False)
    
    if 'dynamic_swin' in args.backbone:
        backbone = DynamicSwinBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif 'swin' in args.backbone:
        backbone = SwinBackbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    else:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

