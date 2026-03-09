#!/usr/bin/env python3
"""
优化的RelTR训练脚本 - 基于收敛分析的改进版本
解决后期收敛慢的问题
"""

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

import os
import sys

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float, help='降低初始学习率')
    parser.add_argument('--lr_backbone', default=5e-6, type=float, help='backbone学习率')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=40, type=int, help='余弦退火周期')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    # 优化参数
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int,
                        help='梯度累积步数，增大有效批大小')
    parser.add_argument('--patience', default=15, type=int,
                        help='早停耐心值')
    parser.add_argument('--scheduler_type', default='cosine', type=str,
                        choices=['cosine', 'plateau', 'step'],
                        help='学习率调度策略')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='预热epoch数')
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model")
    
    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # Transformer
    parser.add_argument('--enc_layers', default=12, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=768, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=12, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,
                        help="Number of object queries")
    parser.add_argument('--rel_num_queries', default=1600, type=int,
                        help="Number of relation queries")
    parser.add_argument('--num_entities', default=800, type=int,
                        help="Number of entity queries used by RelTR")
    parser.add_argument('--num_triplets', default=1600, type=int,
                        help="Number of relation triplet queries used by RelTR")
    parser.add_argument('--pre_norm', action='store_true')
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    # Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="IOU threshold for the matching cost")
    
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1.0, type=float,
                        help="Weight for relation classification loss")
    parser.add_argument('--pointer_loss_coef', default=0.25, type=float,
                        help="Weight for pointer (subject/object) loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    # dataset parameters
    parser.add_argument('--dataset_file', default='pid')
    parser.add_argument('--coco_path', type=str, default='./datasets/pid_dataset')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--pid_path', default='data/pid_resized', type=str,
                        help='Root path for PID dataset (supports resized variant).')
    parser.add_argument('--use_dn', action='store_true',
                        help='Enable denoising training queries.')
    parser.add_argument('--use_deformable_attn', action='store_true',
                        help='Use deformable attention in entity decoder branch.')
    parser.add_argument('--use_grad_checkpoint', action='store_true',
                        help='Use gradient checkpointing in transformer decoder to reduce memory footprint.')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable torch.cuda.amp mixed precision training for speed/memory benefits.')
    parser.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'],
                        help='Autocast dtype when mixed precision is enabled.')
    parser.add_argument('--dn_num_groups', default=2, type=int,
                        help='Number of DN groups when constructing denoising queries.')
    parser.add_argument('--dn_max_gt', default=200, type=int,
                        help='Maximum GT objects replicated into DN queries per image.')
    parser.add_argument('--dn_loss_weight_scale', default=1.0, type=float,
                        help='Global scaling factor for DN losses relative to primary losses.')
    parser.add_argument('--dn_label_noise_ratio', default=0.4, type=float,
                        help='Label noise ratio when generating DN queries.')
    parser.add_argument('--dn_box_noise_scale', default=0.4, type=float,
                        help='BBox noise scale when generating DN queries.')
    parser.add_argument('--num_classes', default=12, type=int,
                        help='Number of entity classes (PID default: 12 without background).')
    parser.add_argument('--num_rel_classes', default=2, type=int,
                        help='Number of relation classes (PID default: 2 without background).')
    
    parser.add_argument('--output_dir', default='./output_optimized',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='./output/checkpoint.pth', type=str, 
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # 添加缺少的参数
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return intermediate layers for multi-scale features")
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    return parser

class OptimizedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
        self.use_amp = bool(getattr(args, "mixed_precision", False) and torch.cuda.is_available())
        if getattr(args, "mixed_precision", False) and not torch.cuda.is_available():
            print("⚠️ Mixed precision requested but CUDA is unavailable; falling back to FP32.")
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        self.amp_dtype = dtype_map.get(getattr(args, "amp_dtype", "float16"), torch.float16)
        self.scaler = GradScaler(enabled=self.use_amp)
        
    def setup_model_and_data(self):
        """设置模型和数据"""
        print("🔧 设置模型和数据加载器...")
        
        # 构建模型
        model, criterion, postprocessors = build_model(self.args)
        model.to(self.device)
        
        # 构建数据集
        dataset_train = build_dataset(image_set='train', args=self.args)
        dataset_val = build_dataset(image_set='val', args=self.args)
        
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, self.args.batch_size, drop_last=True)
        
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                     collate_fn=utils.collate_fn, num_workers=self.args.num_workers)
        data_loader_val = DataLoader(dataset_val, self.args.batch_size, sampler=sampler_val,
                                   drop_last=False, collate_fn=utils.collate_fn, 
                                   num_workers=self.args.num_workers)
        
        return model, criterion, postprocessors, data_loader_train, data_loader_val
    
    def setup_optimizer_and_scheduler(self, model):
        """设置优化器和学习率调度器"""
        print("📊 设置优化器和学习率调度器...")
        
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() 
                       if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": self.args.lr_backbone,
            },
        ]
        
        optimizer = torch.optim.AdamW(param_dicts, lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)
        
        # 设置学习率调度器
        if self.args.scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=self.args.lr_drop, T_mult=2, eta_min=1e-6)
        elif self.args.scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=8, verbose=True)
        else:  # step
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.lr_drop, gamma=0.1)
        
        return optimizer, scheduler
    
    def load_checkpoint(self, model, optimizer, scheduler):
        """加载检查点继续训练"""
        if self.args.resume and os.path.exists(self.args.resume):
            print(f"📂 从检查点继续训练: {self.args.resume}")
            checkpoint = torch.load(self.args.resume, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                self.args.start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                self.best_val_loss = checkpoint['best_val_loss']
            print(f"✅ 从epoch {self.args.start_epoch}继续训练")
        else:
            print("🆕 开始新的训练")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, is_best=False):
        """保存检查点"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'args': self.args,
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.output_dir, 'checkpoint.pth'))
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.output_dir, 'best_checkpoint.pth'))
            print(f"💾 保存最佳模型 (验证损失: {self.best_val_loss:.4f})")
    
    def train_epoch_with_accumulation(self, model, criterion, data_loader, optimizer, epoch):
        """带梯度累积的训练epoch"""
        model.train()
        criterion.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 50
        
        optimizer.zero_grad()
        accumulation_steps = max(1, self.args.gradient_accumulation_steps)
        data_loader_length = len(data_loader)

        for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            samples = samples.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = model(samples, targets)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # 梯度累积
            losses = losses / accumulation_steps
            if self.use_amp:
                self.scaler.scale(losses).backward()
            else:
                losses.backward()

            is_update_step = ((i + 1) % accumulation_steps == 0) or ((i + 1) == data_loader_length)

            if is_update_step:
                if self.args.clip_max_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_max_norm)

                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # 减少loss_dict中的数值
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            
            loss_value = losses_reduced_scaled.item()
            
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            if 'class_error' in loss_dict_reduced:
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
            if 'sub_error' in loss_dict_reduced:
                metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
            if 'obj_error' in loss_dict_reduced:
                metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def train(self):
        """主训练循环"""
        print("🚀 开始优化训练...")
        
        # 设置模型和数据
        model, criterion, postprocessors, data_loader_train, data_loader_val = self.setup_model_and_data()
        
        # 设置优化器和调度器
        optimizer, scheduler = self.setup_optimizer_and_scheduler(model)
        
        # 加载检查点
        self.load_checkpoint(model, optimizer, scheduler)
        
        print(f"📊 训练配置:")
        print(f"  - 初始学习率: {self.args.lr}")
        print(f"  - 梯度累积步数: {self.args.gradient_accumulation_steps}")
        print(f"  - 调度器类型: {self.args.scheduler_type}")
        print(f"  - 早停耐心值: {self.args.patience}")
        print(f"  - 查询数量: {self.args.num_queries}/{self.args.rel_num_queries}")
        print(f"  - Mixed Precision: {'ON' if self.use_amp else 'OFF'}")
        print(f"  - Grad Checkpoint: {'ON' if getattr(self.args, 'use_grad_checkpoint', False) else 'OFF'}")
        
        # 训练循环
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # 训练一个epoch
            train_stats = self.train_epoch_with_accumulation(
                model, criterion, data_loader_train, optimizer, epoch)
            
            # 验证
            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds=None,
                device=self.device,
                args=self.args,
                epoch=epoch,
            )
            
            # 更新学习率
            if self.args.scheduler_type == 'plateau':
                scheduler.step(test_stats['loss'])
            else:
                scheduler.step()
            
            # 记录训练历史
            epoch_stats = {
                'epoch': epoch,
                'train_loss': train_stats['loss'],
                'test_loss': test_stats['loss'],
                'lr': optimizer.param_groups[0]["lr"],
                'train_class_error': train_stats.get('class_error'),
                'train_sub_error': train_stats.get('sub_error'),
                'train_obj_error': train_stats.get('obj_error'),
                'class_error': test_stats.get('class_error', 0),
                'rel_error': test_stats.get('rel_error', 0),
                'sub_error': test_stats.get('sub_error'),
                'obj_error': test_stats.get('obj_error'),
            }
            self.training_history.append(epoch_stats)
            
            # 保存训练日志
            with open(os.path.join(self.args.output_dir, 'training_log_optimized.jsonl'), 'a') as f:
                f.write(json.dumps(epoch_stats) + '\n')
            
            # 检查是否为最佳模型
            current_val_loss = test_stats['loss']
            is_best = current_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(model, optimizer, scheduler, epoch, is_best)
            
            # 早停检查
            if self.patience_counter >= self.args.patience:
                print(f"🛑 早停触发，连续{self.args.patience}个epoch无改善")
                break
            
            # 打印训练状态
            print(f"📊 Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
                  f"val_loss={current_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}, "
                  f"patience={self.patience_counter}/{self.args.patience}")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'🎉 训练完成，总时间: {total_time_str}')
        
        return model

def main(args):
    print("🚀 启动优化的RelTR训练")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 初始化分布式训练
    utils.init_distributed_mode(args)
    
    device = torch.device(args.device)
    
    # 创建训练器并开始训练
    trainer = OptimizedTrainer(args)
    model = trainer.train()
    
    print("✅ 训练完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR优化训练', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Harmonize legacy query flags with RelTR DINOv3 expectations
    if not hasattr(args, 'num_entities') or args.num_entities is None:
        args.num_entities = args.num_queries
    else:
        args.num_queries = args.num_entities

    if not hasattr(args, 'num_triplets') or args.num_triplets is None:
        args.num_triplets = args.rel_num_queries
    else:
        args.rel_num_queries = args.num_triplets

    # Ensure PID dataset path attribute exists for loaders
    if not hasattr(args, 'pid_path') or args.pid_path is None:
        args.pid_path = 'data/pid_resized'

    variant_hidden = {
        'small': (384, 6),
        'base': (768, 12),
        'large': (1024, 16),
    }
    variant = getattr(args, 'dinov3_variant', 'base')
    if variant in variant_hidden:
        hidden_dim, heads = variant_hidden[variant]
        args.hidden_dim = hidden_dim
        args.nheads = heads
    
    main(args)
#!/usr/bin/env python3
"""
优化的训练脚本 - 解决收敛慢的问题
应用收敛分析的改进建议
"""

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from improved_training_config import get_improved_optimizer_and_scheduler, get_improved_training_config, apply_warmup_schedule, get_dynamic_queries
import sys
import os

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=5e-5, type=float)  # 降低初始学习率
    parser.add_argument('--lr_backbone', default=5e-6, type=float)  # backbone更小学习率
    parser.add_argument('--lr_scheduler', default='plateau', help='学习率调度策略')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)  # 增加总epochs
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dinov3_variant', default='base', type=str,
                        choices=['small', 'base', 'large'],
                        help='Choose DINOv3 backbone variant (controls channel dim).')
    parser.add_argument('--dinov3_patch_size', default=16, type=int,
                        help='Patch size for DINOv3 backbone.')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=800, type=int,  # 使用800查询
                        help="Number of query slots")
    parser.add_argument('--num_rel_queries', default=1600, type=int,  # 使用1600关系查询
                        help="Number of relation query slots")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='pid')
    parser.add_argument('--pid_path', default='data/pid', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='optimized_output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # 新增优化参数
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int,
                        help='梯度累积步数')
    parser.add_argument('--early_stopping_patience', default=15, type=int,
                        help='早停耐心值')
    parser.add_argument('--save_best_only', default=True, type=bool,
                        help='只保存最佳模型')
    
    return parser

class EarlyStopping:
    """早停类"""
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 获取改进的优化器和调度器
    optimizer, lr_scheduler = get_improved_optimizer_and_scheduler(model_without_ddp, args)
    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        coco_val = dataset_val.coco
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args,
        )
        if args.output_dir:
            utils.save_on_master(test_stats, output_dir / "eval.pth")
        return

    # 获取改进的训练配置
    config = get_improved_training_config()
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=args.early_stopping_patience)
    
    # 训练日志
    training_log = []
    best_val_loss = float('inf')
    
    print("🚀 开始优化训练...")
    print(f"📋 配置: LR={args.lr}, 调度器={args.lr_scheduler}, 累积步数={args.gradient_accumulation_steps}")
    
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        # 应用warmup
        if epoch < config['warmup_epochs']:
            apply_warmup_schedule(optimizer, epoch, config['warmup_epochs'], args.lr)
            
        # 动态调整查询数量
        if config['dynamic_queries']:
            current_queries = get_dynamic_queries(epoch, config)
            if current_queries:
                args.num_queries, args.num_rel_queries = current_queries
                args.num_entities, args.num_triplets = current_queries
                print(f"📊 Epoch {epoch}: 调整查询数量到 {args.num_queries}/{args.num_rel_queries}")

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, accumulation_steps=args.gradient_accumulation_steps)
            
        if args.lr_scheduler == 'plateau':
            # 对于ReduceLROnPlateau，我们需要传入验证损失
            val_stats = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args,
                epoch,
            )
            lr_scheduler.step(val_stats['loss'])
            val_loss = val_stats['loss']
        else:
            lr_scheduler.step()
            val_stats = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args,
                epoch,
            )
            val_loss = val_stats['loss']

        # 记录训练日志
        log_entry = {
            'epoch': epoch,
            'train_loss': train_stats['loss'],
            'test_loss': val_loss,
            'train_class_error': train_stats.get('class_error', 0),
            'test_class_error': val_stats.get('class_error', 0),
            'train_rel_error': train_stats.get('rel_error', 0),
            'test_rel_error': val_stats.get('rel_error', 0),
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)
        
        # 打印进度
        print(f"Epoch {epoch}: train_loss={train_stats['loss']:.4f}, "
              f"val_loss={val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_paths.append(output_dir / 'checkpoint_best.pth')
                print(f"💾 保存最佳模型，val_loss: {val_loss:.4f}")
            
            # 只在改进时保存或者每10个epoch保存一次
            if args.save_best_only:
                if val_loss < best_val_loss or epoch % 10 == 0:
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
            else:
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

        # 保存训练日志
        with open(output_dir / 'training_log_optimized.json', 'w') as f:
            for entry in training_log:
                f.write(json.dumps(entry) + '\n')
                
        # 早停检查
        if early_stopping(val_loss):
            print(f"🛑 早停在epoch {epoch}，连续{args.early_stopping_patience}个epoch无改善")
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not hasattr(args, 'num_entities') or args.num_entities is None:
        args.num_entities = getattr(args, 'num_queries', 800)
    else:
        args.num_queries = args.num_entities

    if not hasattr(args, 'num_triplets') or args.num_triplets is None:
        args.num_triplets = getattr(args, 'rel_num_queries', 1600)
    else:
        args.rel_num_queries = args.num_triplets

    if not hasattr(args, 'pid_path') or args.pid_path is None:
        args.pid_path = 'data/pid_resized'

    # Adjust transformer dimensions to match DINOv3 variant
    variant_hidden = {
        'small': (384, 6),
        'base': (768, 12),
        'large': (1024, 16),
    }
    variant = getattr(args, 'dinov3_variant', 'base')
    if variant in variant_hidden:
        hidden_dim, heads = variant_hidden[variant]
        args.hidden_dim = hidden_dim
        args.nheads = heads

    main(args)
