#!/usr/bin/env python3
"""
简化的PID2Graph损失函数
直接处理decoder输出的索引预测，无需复杂匹配器
"""

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import accuracy


class SimplePID2GraphCriterion(nn.Module):
    """
    简化的PID2Graph损失函数
    
    直接处理decoder的输出：
    - pred_logits: [B, num_entities, num_classes+1]
    - pred_boxes: [B, num_entities, 4] 
    - rel_logits: [B, num_triplets, num_rel_classes+1]
    - subject_indices: [B, num_triplets, num_entities] 主体索引预测
    - object_indices: [B, num_triplets, num_entities] 客体索引预测
    
    目标格式（PID2Graph）：
    - boxes: [N, 4] 实体边界框
    - labels: [N] 实体标签
    - rel_annotations: [M, 3] [主体索引, 客体索引, 关系标签]
    """
    
    def __init__(self, num_classes, num_rel_classes, weight_dict, 
                 eos_coef=0.1, losses=['labels', 'boxes', 'relations']):
        """
        Args:
            num_classes: 实体类别数（不包括背景）
            num_rel_classes: 关系类别数（不包括无关系）
            weight_dict: 损失权重字典
            eos_coef: 背景类权重
            losses: 要计算的损失类型列表
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
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
        
        print(f"🔧 初始化简化PID2Graph损失函数:")
        print(f"   实体类别数: {num_classes}")
        print(f"   关系类别数: {num_rel_classes}")
        print(f"   损失类型: {losses}")

    def loss_labels(self, outputs, targets, num_boxes, log=True):
        """
        计算实体分类损失
        使用简单的前K个预测与前K个目标匹配
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # [B, num_entities, num_classes+1]
        
        # 收集所有目标
        all_target_labels = []
        all_target_valid = []
        
        for batch_idx, target in enumerate(targets):
            target_labels = target["labels"]  # [N]
            num_targets = len(target_labels)
            
            # 创建填充目标（用背景类填充到num_entities长度）
            padded_labels = torch.full((src_logits.shape[1],), self.num_classes, 
                                     dtype=torch.int64, device=src_logits.device)
            padded_labels[:num_targets] = target_labels
            
            # 创建有效性掩码
            valid_mask = torch.zeros(src_logits.shape[1], dtype=torch.bool, device=src_logits.device)
            valid_mask[:num_targets] = True
            
            all_target_labels.append(padded_labels)
            all_target_valid.append(valid_mask)
        
        target_classes = torch.stack(all_target_labels)  # [B, num_entities]
        valid_masks = torch.stack(all_target_valid)      # [B, num_entities]
        
        # 计算分类损失（只对有效位置）
        # 确保权重在正确的设备上
        weight = self.empty_weight.to(src_logits.device)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, 
                                 weight, reduction='none')  # [B, num_entities]
        
        # 只对有效位置计算损失
        loss_ce = loss_ce * valid_masks.float()
        loss_ce = loss_ce.sum() / valid_masks.sum().clamp(min=1)
        
        losses = {'loss_ce': loss_ce}

        if log:
            # 计算准确率（只对有效位置）
            pred_classes = src_logits.argmax(-1)  # [B, num_entities]
            correct = (pred_classes == target_classes) * valid_masks
            accuracy_val = correct.sum().float() / valid_masks.sum().clamp(min=1)
            losses['class_error'] = 100 - accuracy_val * 100
            
        return losses

    def loss_boxes(self, outputs, targets, num_boxes):
        """计算边界框回归损失"""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']  # [B, num_entities, 4]
        
        # 收集所有目标边界框
        all_target_boxes = []
        all_target_valid = []
        
        for batch_idx, target in enumerate(targets):
            target_boxes = target['boxes']  # [N, 4]
            num_targets = len(target_boxes)
            
            # 创建填充目标（用零填充到num_entities长度）
            padded_boxes = torch.zeros((src_boxes.shape[1], 4), device=src_boxes.device)
            padded_boxes[:num_targets] = target_boxes
            
            # 创建有效性掩码
            valid_mask = torch.zeros(src_boxes.shape[1], dtype=torch.bool, device=src_boxes.device)
            valid_mask[:num_targets] = True
            
            all_target_boxes.append(padded_boxes)
            all_target_valid.append(valid_mask)
        
        target_boxes = torch.stack(all_target_boxes)  # [B, num_entities, 4]
        valid_masks = torch.stack(all_target_valid)   # [B, num_entities]
        
        # 只对有效位置计算L1损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')  # [B, num_entities, 4]
        loss_bbox = loss_bbox.sum(-1) * valid_masks.float()  # [B, num_entities]
        loss_bbox = loss_bbox.sum() / valid_masks.sum().clamp(min=1)
        
        # 计算GIoU损失
        valid_src_boxes = src_boxes[valid_masks]      # [num_valid, 4]
        valid_target_boxes = target_boxes[valid_masks] # [num_valid, 4]
        
        if len(valid_src_boxes) > 0:
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(valid_src_boxes),
                box_ops.box_cxcywh_to_xyxy(valid_target_boxes)))
            loss_giou = loss_giou.mean()
        else:
            loss_giou = torch.tensor(0.0, device=src_boxes.device)
            
        losses = {
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
        return losses

    def loss_relations(self, outputs, targets, num_boxes, log=True):
        """
        计算关系损失
        包括关系分类、主体索引预测、客体索引预测
        """
        assert 'rel_logits' in outputs
        assert 'subject_indices' in outputs  
        assert 'object_indices' in outputs
        
        rel_logits = outputs['rel_logits']        # [B, num_triplets, num_rel_classes+1]
        subject_indices = outputs['subject_indices']  # [B, num_triplets, num_entities]
        object_indices = outputs['object_indices']    # [B, num_triplets, num_entities]
        
        # 收集所有目标关系
        all_target_rels = []
        all_target_subjects = []
        all_target_objects = []
        all_rel_valid = []
        
        for batch_idx, target in enumerate(targets):
            rel_annotations = target["rel_annotations"]  # [M, 3] = [sub_idx, obj_idx, rel_label]
            num_rels = len(rel_annotations)
            
            # 创建填充目标
            padded_rels = torch.full((rel_logits.shape[1],), self.num_rel_classes, 
                                   dtype=torch.int64, device=rel_logits.device)
            padded_subjects = torch.zeros(rel_logits.shape[1], dtype=torch.int64, device=rel_logits.device)
            padded_objects = torch.zeros(rel_logits.shape[1], dtype=torch.int64, device=rel_logits.device)
            
            if num_rels > 0:
                padded_rels[:num_rels] = rel_annotations[:, 2]      # 关系标签
                padded_subjects[:num_rels] = rel_annotations[:, 0]  # 主体索引
                padded_objects[:num_rels] = rel_annotations[:, 1]   # 客体索引
            
            # 创建有效性掩码
            valid_mask = torch.zeros(rel_logits.shape[1], dtype=torch.bool, device=rel_logits.device)
            valid_mask[:num_rels] = True
            
            all_target_rels.append(padded_rels)
            all_target_subjects.append(padded_subjects)
            all_target_objects.append(padded_objects)
            all_rel_valid.append(valid_mask)
        
        target_rels = torch.stack(all_target_rels)      # [B, num_triplets]
        target_subjects = torch.stack(all_target_subjects)  # [B, num_triplets]
        target_objects = torch.stack(all_target_objects)    # [B, num_triplets]
        rel_valid_masks = torch.stack(all_rel_valid)        # [B, num_triplets]
        
        # 关系分类损失
        weight_rel = self.empty_weight_rel.to(rel_logits.device)
        loss_rel_ce = F.cross_entropy(rel_logits.transpose(1, 2), target_rels, 
                                     weight_rel, reduction='none')
        loss_rel_ce = loss_rel_ce * rel_valid_masks.float()
        loss_rel_ce = loss_rel_ce.sum() / rel_valid_masks.sum().clamp(min=1)
        
        # 主体索引预测损失
        loss_subject = F.cross_entropy(subject_indices.transpose(1, 2), target_subjects, reduction='none')
        loss_subject = loss_subject * rel_valid_masks.float()
        loss_subject = loss_subject.sum() / rel_valid_masks.sum().clamp(min=1)
        
        # 客体索引预测损失
        loss_object = F.cross_entropy(object_indices.transpose(1, 2), target_objects, reduction='none')
        loss_object = loss_object * rel_valid_masks.float()
        loss_object = loss_object.sum() / rel_valid_masks.sum().clamp(min=1)
        
        losses = {
            'loss_rel': loss_rel_ce,
            'loss_subject': loss_subject,
            'loss_object': loss_object
        }

        if log:
            # 计算准确率
            pred_rels = rel_logits.argmax(-1)
            pred_subjects = subject_indices.argmax(-1)  
            pred_objects = object_indices.argmax(-1)
            
            rel_correct = (pred_rels == target_rels) * rel_valid_masks
            subj_correct = (pred_subjects == target_subjects) * rel_valid_masks
            obj_correct = (pred_objects == target_objects) * rel_valid_masks
            
            valid_count = rel_valid_masks.sum().clamp(min=1)
            losses['rel_error'] = 100 - (rel_correct.sum().float() / valid_count) * 100
            losses['subject_error'] = 100 - (subj_correct.sum().float() / valid_count) * 100
            losses['object_error'] = 100 - (obj_correct.sum().float() / valid_count) * 100
            
        return losses

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        """获取指定类型的损失"""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        前向传播计算损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标列表
                
        Returns:
            损失字典
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # 计算所有目标的数量（用于归一化）
        num_boxes = sum(len(t["labels"]) + len(t["rel_annotations"]) for t in targets)
        num_boxes = max(num_boxes, 1)  # 避免除零

        # 计算所有损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs_without_aux, targets, num_boxes))

        # 处理辅助损失（如果存在）
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    elif loss == 'relations':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_simple_pid2graph_criterion(args):
    """构建简化的PID2Graph格式损失函数"""
    
    # 损失权重
    weight_dict = {
        'loss_ce': getattr(args, 'cls_loss_coef', 1),
        'loss_bbox': getattr(args, 'bbox_loss_coef', 5),
        'loss_giou': getattr(args, 'giou_loss_coef', 2),
        'loss_rel': getattr(args, 'rel_loss_coef', 1),
        'loss_subject': getattr(args, 'subject_loss_coef', 1),
        'loss_object': getattr(args, 'object_loss_coef', 1),
    }
    
    # 辅助损失权重
    if getattr(args, 'aux_loss', False):
        aux_weight_dict = {}
        for i in range(getattr(args, 'dec_layers', 6) - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    losses = ['labels', 'boxes', 'relations']
    
    criterion = SimplePID2GraphCriterion(
        num_classes=getattr(args, 'num_classes', 91),
        num_rel_classes=getattr(args, 'num_rel_classes', 50),
        weight_dict=weight_dict,
        eos_coef=getattr(args, 'eos_coef', 0.1),
        losses=losses
    )
    
    print("✅ 简化PID2Graph损失函数构建完成")
    return criterion, weight_dict
