#!/usr/bin/env python3
"""
PID2Graph格式兼容的损失函数
为用户设计的架构提供训练支持，支持rel_annotations格式
"""

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized


class PID2GraphCriterion(nn.Module):
    """
    专门为PID2Graph格式设计的损失函数
    
    输入格式：
    - pred_logits: [B, num_entities, num_classes+1]
    - pred_boxes: [B, num_entities, 4] 
    - rel_logits: [B, num_triplets, num_rel_classes+1]
    
    目标格式（PID2Graph）：
    - boxes: [N, 4] 实体边界框
    - labels: [N] 实体标签
    - rel_annotations: [M, 3] [主体索引, 客体索引, 关系标签]
    """
    
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, 
                 eos_coef=0.1, losses=['labels', 'boxes', 'relations']):
        """
        Args:
            num_classes: 实体类别数（不包括背景）
            num_rel_classes: 关系类别数（不包括无关系）
            matcher: 匈牙利匹配器
            weight_dict: 损失权重字典
            eos_coef: 背景类权重
            losses: 要计算的损失类型列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # 为实体分类创建权重（降低背景类权重）
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        
        # 为关系分类创建权重
        empty_weight_rel = torch.ones(self.num_rel_classes + 1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)
        
        print(f"🔧 初始化PID2Graph损失函数:")
        print(f"   实体类别数: {num_classes}")
        print(f"   关系类别数: {num_rel_classes}")
        print(f"   损失类型: {losses}")

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """计算实体分类损失"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """计算基数误差（仅用于日志记录）"""
        # 实体基数误差
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        
        # 关系基数误差
        rel_logits = outputs['rel_logits']
        rel_tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        rel_card_pred = (rel_logits.argmax(-1) != rel_logits.shape[-1] - 1).sum(1)
        rel_card_err = F.l1_loss(rel_card_pred.float(), rel_tgt_lengths.float())
        
        losses = {
            'cardinality_error': card_err,
            'rel_cardinality_error': rel_card_err
        }
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """计算边界框回归损失"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """
        计算关系分类损失
        
        这里我们需要处理PID2Graph格式的rel_annotations：
        rel_annotations: [M, 3] 其中每行是 [主体索引, 客体索引, 关系标签]
        """
        assert 'rel_logits' in outputs
        src_logits = outputs['rel_logits']
        
        # 获取关系匹配索引
        idx = self._get_src_permutation_idx(indices[1])
        
        # 从rel_annotations中提取关系标签
        target_classes_o = torch.cat([t["rel_annotations"][J, 2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)
        losses = {'loss_rel': loss_ce}

        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        """获取源排列索引"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """获取目标排列索引"""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """获取指定类型的损失"""
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        前向传播计算损失
        
        Args:
            outputs: 模型输出字典
                - pred_logits: [B, num_entities, num_classes+1]
                - pred_boxes: [B, num_entities, 4]
                - rel_logits: [B, num_triplets, num_rel_classes+1]
                - aux_outputs: 辅助输出列表（可选）
            targets: 目标列表，每个目标包含：
                - boxes: [N, 4] 
                - labels: [N]
                - rel_annotations: [M, 3]
                
        Returns:
            损失字典
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 使用匈牙利匹配器计算最佳匹配
        # 我们需要分别匹配实体和关系
        entity_indices = self.matcher(outputs_without_aux, targets, match_type='entity')
        rel_indices = self.matcher(outputs_without_aux, targets, match_type='relation')
        
        indices = [entity_indices, rel_indices]

        # 计算所有节点的数量（用于归一化）
        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 处理辅助损失（如果存在）
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_entity_indices = self.matcher(aux_outputs, targets, match_type='entity')
                aux_rel_indices = self.matcher(aux_outputs, targets, match_type='relation')
                aux_indices = [aux_entity_indices, aux_rel_indices]
                
                for loss in self.losses:
                    if loss == 'cardinality':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, aux_indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_pid2graph_criterion(args):
    """构建PID2Graph格式的损失函数"""
    from models.pid2graph_matcher import PID2GraphMatcher
    
    # 创建专门的匹配器
    matcher = PID2GraphMatcher(
        cost_class=getattr(args, 'set_cost_class', 1),
        cost_bbox=getattr(args, 'set_cost_bbox', 5),
        cost_giou=getattr(args, 'set_cost_giou', 2),
        cost_rel=getattr(args, 'set_cost_rel', 1)
    )
    
    # 损失权重
    weight_dict = {
        'loss_ce': getattr(args, 'cls_loss_coef', 1),
        'loss_bbox': getattr(args, 'bbox_loss_coef', 5),
        'loss_giou': getattr(args, 'giou_loss_coef', 2),
        'loss_rel': getattr(args, 'rel_loss_coef', 1),
    }
    
    # 辅助损失权重
    if getattr(args, 'aux_loss', False):
        aux_weight_dict = {}
        for i in range(getattr(args, 'dec_layers', 6) - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'cardinality', 'relations']
    
    criterion = PID2GraphCriterion(
        num_classes=getattr(args, 'num_classes', 91),
        num_rel_classes=getattr(args, 'num_rel_classes', 50),
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=getattr(args, 'eos_coef', 0.1),
        losses=losses
    )
    
    print("✅ PID2Graph损失函数构建完成")
    return criterion, weight_dict
