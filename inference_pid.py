#!/usr/bin/env python3
"""
PID 推理脚本 - 使用训练好的 RelTRv3 模型进行场景图生成
"""

import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入模型和数据集相关模块
import sys
sys.path.append('.')

# build_reltr will be imported lazily in load_model to avoid heavy module imports at import time
from datasets.pid_datasets import build_pid_dataset
from datasets.transforms import Compose, Normalize, ToTensor
from util.misc import nested_tensor_from_tensor_list
import util.misc as utils

# PID数据集的类别映射 (简化版本，与训练时匹配)
PID_CLASSES = [
    'background', 'person', 'bicycle', 'chair', 'table', 'handbag', 'laptop'
]

PID_PREDICATES = [
    'background', 'hold', 'sit_on'
]

def get_args_parser():
    parser = argparse.ArgumentParser('PID Inference', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    
    # PID dataset parameters (adjusted to match actual training)
    parser.add_argument('--num_classes', default=7, type=int,
                        help="Number of object classes (excluding background)")
    parser.add_argument('--num_rel_classes', default=2, type=int,
                        help="Number of relation classes (excluding background)")
    parser.add_argument('--num_entities', default=800, type=int,
                        help="Number of entity queries") 
    parser.add_argument('--num_triplets', default=800, type=int,
                        help="Number of triplet queries")
    
    # Transformer parameters
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pre_norm', action='store_true')
    
    # Matcher parameters
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="IoU threshold for matching")
    
    # Loss coefficients (needed for model building)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    # Additional model parameters for building
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help="Learning rate for backbone (used in model building)")
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--aux_loss', default=True, type=bool,
                        help="auxiliary decoding losses (loss at each layer)")
    
    # Dataset parameters  
    parser.add_argument('--dataset_file', default='pid')
    parser.add_argument('--data_path', type=str, default='data/pid_resized/')
    parser.add_argument('--remove_difficult', action='store_true')
    
    # Inference parameters
    parser.add_argument('--model_path', required=True, type=str,
                        help="Path to the trained model checkpoint")
    parser.add_argument('--image_path', required=True, type=str,
                        help="Path to the input image")
    parser.add_argument('--output_dir', default='inference_results', type=str,
                        help="Output directory for results")
    parser.add_argument('--confidence_threshold', default=0.5, type=float,
                        help="Confidence threshold for predictions")
    parser.add_argument('--max_detections', default=100, type=int,
                        help="Maximum number of detections")
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device to use (cuda or cpu)")
    
    return parser

def load_model(args):
    """加载训练好的模型"""
    print(f"🔄 加载模型从: {args.model_path}")
    
    device = torch.device(args.device)
    
    # 构建模型（延迟导入以避免在脚本导入时触发大量依赖）
    from importlib import import_module
    build_fn = import_module('models.reltr').build_reltr
    model, _, _ = build_fn(args)
    model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"✅ 模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    return model, device

def preprocess_image(image_path):
    """预处理输入图像"""
    print(f"🔄 预处理图像: {image_path}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    orig_size = torch.tensor([image.height, image.width])
    
    # 应用变换
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = Compose([
        ToTensor(),
        normalize,
    ])
    transformed_image, _ = transform(image, None)
    
    # 创建nested tensor
    samples = nested_tensor_from_tensor_list([transformed_image])
    
    print(f"✅ 图像预处理完成，原始尺寸: {image.size}, 变换后尺寸: {transformed_image.shape}")
    return samples, orig_size, image

def postprocess_predictions(outputs, orig_size, confidence_threshold=0.5, max_detections=100):
    """后处理模型预测结果"""
    print(f"🔄 后处理预测结果，置信度阈值: {confidence_threshold}")
    
    # 获取预测结果
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]
    sub_logits = outputs['sub_logits'][0]    # [num_queries, num_classes] 
    obj_logits = outputs['obj_logits'][0]    # [num_queries, num_classes]
    rel_logits = outputs['rel_logits'][0]    # [num_queries, num_predicates]
    
    # 获取边界框预测
    pred_scores = pred_logits.softmax(-1)[:, :-1]  # 排除背景类
    pred_labels = pred_scores.argmax(-1)
    pred_scores = pred_scores.max(-1)[0]
    
    # 过滤低置信度检测
    keep = pred_scores > confidence_threshold
    if keep.sum() == 0:
        print("⚠️ 没有找到置信度足够高的检测结果")
        return [], []
    
    # 限制检测数量
    if keep.sum() > max_detections:
        top_scores, top_indices = pred_scores[keep].topk(max_detections)
        keep_indices = torch.where(keep)[0][top_indices]
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[keep_indices] = True
    
    # 提取有效检测
    valid_boxes = pred_boxes[keep]
    valid_labels = pred_labels[keep]
    valid_scores = pred_scores[keep]
    
    # 转换边界框坐标到原始图像尺寸
    # boxes 格式: [cx, cy, w, h] (归一化) -> [x1, y1, x2, y2] (像素)
    h, w = orig_size
    valid_boxes[:, [0, 2]] *= w  # cx, w
    valid_boxes[:, [1, 3]] *= h  # cy, h
    
    # 转换为 x1, y1, x2, y2 格式
    boxes_x1y1x2y2 = torch.zeros_like(valid_boxes)
    boxes_x1y1x2y2[:, 0] = valid_boxes[:, 0] - valid_boxes[:, 2] / 2  # x1
    boxes_x1y1x2y2[:, 1] = valid_boxes[:, 1] - valid_boxes[:, 3] / 2  # y1
    boxes_x1y1x2y2[:, 2] = valid_boxes[:, 0] + valid_boxes[:, 2] / 2  # x2
    boxes_x1y1x2y2[:, 3] = valid_boxes[:, 1] + valid_boxes[:, 3] / 2  # y2
    
    # 处理关系预测
    sub_scores = sub_logits[keep].softmax(-1)[:, :-1]
    obj_scores = obj_logits[keep].softmax(-1)[:, :-1] 
    rel_scores = rel_logits[keep].softmax(-1)[:, :-1]  # 排除背景关系
    
    # 构建检测结果
    detections = []
    for i in range(len(valid_boxes)):
        detection = {
            'bbox': boxes_x1y1x2y2[i].cpu().numpy(),
            'label': valid_labels[i].item(),
            'class_name': PID_CLASSES[valid_labels[i].item()],
            'confidence': valid_scores[i].item()
        }
        detections.append(detection)
    
    # 构建关系结果（简化版本，选择最高置信度的关系）
    relations = []
    for i in range(len(valid_boxes)):
        for j in range(len(valid_boxes)):
            if i != j:
                # 使用对应的关系分数
                rel_score = rel_scores[i].max().item()
                rel_pred = rel_scores[i].argmax().item()
                
                if rel_score > confidence_threshold:
                    relation = {
                        'subject': i,
                        'object': j,
                        'predicate': rel_pred,
                        'predicate_name': PID_PREDICATES[rel_pred] if rel_pred < len(PID_PREDICATES) else f'rel_{rel_pred}',
                        'confidence': rel_score
                    }
                    relations.append(relation)
    
    print(f"✅ 后处理完成，检测到 {len(detections)} 个对象，{len(relations)} 个关系")
    return detections, relations

def visualize_results(image, detections, relations, output_path):
    """可视化检测和关系结果"""
    print(f"🔄 生成可视化结果")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    # 绘制边界框和标签
    colors = plt.cm.Set3(np.linspace(0, 1, len(detections)))
    
    for i, det in enumerate(detections):
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        
        # 添加标签
        label_text = f"{det['class_name']}: {det['confidence']:.2f}"
        ax.text(x1, y1-5, label_text, fontsize=10, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7))
    
    # 绘制关系连线
    for rel in relations:
        if rel['subject'] < len(detections) and rel['object'] < len(detections):
            subj_bbox = detections[rel['subject']]['bbox']
            obj_bbox = detections[rel['object']]['bbox']
            
            # 计算边界框中心点
            subj_center = [(subj_bbox[0] + subj_bbox[2])/2, (subj_bbox[1] + subj_bbox[3])/2]
            obj_center = [(obj_bbox[0] + obj_bbox[2])/2, (obj_bbox[1] + obj_bbox[3])/2]
            
            # 绘制连线
            ax.annotate('', xy=obj_center, xytext=subj_center,
                       arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
            
            # 添加关系标签
            mid_x = (subj_center[0] + obj_center[0]) / 2
            mid_y = (subj_center[1] + obj_center[1]) / 2
            ax.text(mid_x, mid_y, rel['predicate_name'], 
                   fontsize=8, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    ax.set_title('PID Scene Graph Detection Results', fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 可视化结果保存到: {output_path}")

def save_results(detections, relations, output_path):
    """保存检测结果为JSON格式"""
    
    # 转换numpy数组为列表，确保JSON可序列化
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results = {
        'detections': convert_to_serializable(detections),
        'relations': convert_to_serializable(relations),
        'num_detections': len(detections),
        'num_relations': len(relations)
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 检测结果保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser('PID Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    
    print("🚀 开始PID场景图生成推理")
    print(f"   模型路径: {args.model_path}")
    print(f"   图像路径: {args.image_path}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   设备: {args.device}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 加载模型
    model, device = load_model(args)
    
    # 预处理图像
    samples, orig_size, orig_image = preprocess_image(args.image_path)
    samples = samples.to(device)
    
    # 推理
    print("🔄 执行模型推理...")
    with torch.no_grad():
        outputs = model(samples)
    
    # 后处理
    detections, relations = postprocess_predictions(
        outputs, orig_size, 
        confidence_threshold=args.confidence_threshold,
        max_detections=args.max_detections
    )
    
    # 保存和可视化结果
    image_name = Path(args.image_path).stem
    
    # 保存JSON结果
    json_path = output_dir / f"{image_name}_results.json"
    save_results(detections, relations, json_path)
    
    # 生成可视化
    viz_path = output_dir / f"{image_name}_visualization.png"
    visualize_results(orig_image, detections, relations, viz_path)
    
    # 打印摘要
    print("\n📊 推理结果摘要:")
    print(f"   检测到对象数量: {len(detections)}")
    print(f"   检测到关系数量: {len(relations)}")
    
    if detections:
        print("\n🎯 检测到的对象:")
        for i, det in enumerate(detections[:10]):  # 只显示前10个
            print(f"   {i+1}. {det['class_name']} (置信度: {det['confidence']:.3f})")
        if len(detections) > 10:
            print(f"   ... 还有 {len(detections)-10} 个对象")
    
    if relations:
        print("\n🔗 检测到的关系:")
        for i, rel in enumerate(relations[:10]):  # 只显示前10个
            if rel['subject'] < len(detections) and rel['object'] < len(detections):
                subj_name = detections[rel['subject']]['class_name']
                obj_name = detections[rel['object']]['class_name']
                print(f"   {i+1}. {subj_name} -> {rel['predicate_name']} -> {obj_name} (置信度: {rel['confidence']:.3f})")
        if len(relations) > 10:
            print(f"   ... 还有 {len(relations)-10} 个关系")
    
    print(f"\n✅ 推理完成！结果保存在: {output_dir}")

if __name__ == '__main__':
    main()
