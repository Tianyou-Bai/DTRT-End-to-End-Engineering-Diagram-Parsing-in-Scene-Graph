#!/usr/bin/env python3
"""
PID2Graph格式的匈牙利匹配器
支持实体和关系的分离匹配
"""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class PID2GraphMatcher(nn.Module):
    """
    PID2Graph格式的匈牙利匹配器
    
    分别处理实体匹配和关系匹配：
    - 实体匹配：基于分类和边界框
    - 关系匹配：基于关系分类和参与实体
    """
    
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, 
                 cost_giou: float = 1, cost_rel: float = 1):
        """
        Args:
            cost_class: 实体分类成本权重
            cost_bbox: 边界框L1成本权重  
            cost_giou: 边界框GIoU成本权重
            cost_rel: 关系分类成本权重
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_rel = cost_rel
        
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0 or cost_rel != 0, \
            "所有成本不能都为0"
            
        print(f"🔧 初始化PID2Graph匹配器:")
        print(f"   实体分类成本: {cost_class}")
        print(f"   边界框L1成本: {cost_bbox}")
        print(f"   边界框GIoU成本: {cost_giou}")
        print(f"   关系分类成本: {cost_rel}")

    @torch.no_grad()
    def forward(self, outputs, targets, match_type='entity'):
        """
        执行匹配
        
        Args:
            outputs: 模型输出
            targets: 目标列表
            match_type: 'entity' 或 'relation'
            
        Returns:
            匹配的索引列表
        """
        if match_type == 'entity':
            return self._match_entities(outputs, targets)
        elif match_type == 'relation':
            return self._match_relations(outputs, targets)
        else:
            raise ValueError(f"未知的匹配类型: {match_type}")

    def _match_entities(self, outputs, targets):
        """匹配实体"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 扁平化以进行匹配
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 连接所有目标标签和边界框
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 计算分类成本
        cost_class = -out_prob[:, tgt_ids]

        # 计算L1成本
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 计算GIoU成本
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))

        # 最终成本矩阵
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]

    def _match_relations(self, outputs, targets):
        """匹配关系"""
        bs, num_queries = outputs["rel_logits"].shape[:2]

        # 扁平化关系预测
        out_rel_prob = outputs["rel_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_rel_classes]

        # 连接所有目标关系标签
        tgt_rel_ids = torch.cat([v["rel_annotations"][:, 2] for v in targets])  # 取关系标签

        # 计算关系分类成本
        cost_rel = -out_rel_prob[:, tgt_rel_ids]

        # 最终成本矩阵
        C = self.cost_rel * cost_rel
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["rel_annotations"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


def build_pid2graph_matcher(args):
    """构建PID2Graph匹配器"""
    return PID2GraphMatcher(
        cost_class=getattr(args, 'set_cost_class', 1),
        cost_bbox=getattr(args, 'set_cost_bbox', 5),
        cost_giou=getattr(args, 'set_cost_giou', 2),
        cost_rel=getattr(args, 'set_cost_rel', 1)
    )
