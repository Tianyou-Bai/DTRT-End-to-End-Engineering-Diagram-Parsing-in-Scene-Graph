#!/usr/bin/env python3
"""
RelTR 训练脚本 - 针对调整分辨率的PID数据集优化
使用1024像素的调整后图像进行高效训练
"""

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('RelTR Training', add_help=False)
    
    # 学习率和优化器
    parser.add_argument('--lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='backbone学习率')
    parser.add_argument('--batch_size', default=2, type=int, help='批次大小 (适配1024像素图像)')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr_drop', default=150, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='梯度裁剪的最大范数')

    # 模型参数
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="冻结预训练权重的路径")

    # * 损失参数
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="禁用辅助损失")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="相对分类权重")
    
    # * 模型架构
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="transformer解码器层数")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="transformer编码器层数")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="transformer中前馈层的中间维度")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="transformer嵌入维度")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="transformer中的dropout")
    parser.add_argument('--nheads', default=8, type=int,
                        help="注意力头数")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="位置编码类型")
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="用于backbone的CNN模型")
    parser.add_argument('--dilation', action='store_true',
                        help="启用backbone最后一个卷积块的膨胀")
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="如果有标签则返回fpn")
                        
    # * DINO / CDN 参数
    parser.add_argument('--dn_number', default=5, type=int, help='Denoising group number')
    parser.add_argument('--dn_box_noise_scale', default=0.4, type=float, help='Denoising box noise scale')
    parser.add_argument('--dn_label_noise_ratio', default=0.5, type=float, help='Denoising label noise ratio')

    
    # * Matcher参数
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="匹配成本中的类别系数")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="匹配成本中的L1框系数")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="匹配成本中的giou框系数")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="IoU阈值")
    
    # * RelTR配置
    parser.add_argument('--num_entities', default=800, type=int,
                        help="实体查询数量")
    parser.add_argument('--num_triplets', default=1600, type=int,
                        help="关系三元组查询数量")

    # * 数据集参数
    parser.add_argument('--dataset_file', default='pid', help='数据集名称')
    parser.add_argument('--pid_path', default='data/pid_resized', 
                        help='PID数据集路径 (使用调整分辨率后的数据)')
    parser.add_argument('--remove_difficult', action='store_true')

    # * 动态剪枝参数
    parser.add_argument('--enable_reconstruction', action='store_true',
                        help='Enable auxiliary reconstruction head training')
    parser.add_argument('--recon_loss_coef', default=1.0, type=float,
                        help='Coefficient for reconstruction loss')


    # * 训练配置
    parser.add_argument('--output_dir', default='./outputs/pid_1024_training',
                        help='保存模型和日志的目录')
    parser.add_argument('--device', default='cuda',
                        help='训练设备 (default: cuda)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='从检查点恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='起始epoch')
    parser.add_argument('--eval', action='store_true', help='仅评估模式')
    parser.add_argument('--num_workers', default=2, type=int)

    # * 评估设置
    parser.add_argument('--eval_every', default=10, type=int, 
                        help='每N个epoch评估一次')
    parser.add_argument('--save_every', default=20, type=int,
                        help='每N个epoch保存检查点')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # 设置随机种子
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 手动设置PID数据集的类别数量
    # 对象类别: arrow, connector, crossing, general, instrumentation, valve (6类)
    args.num_classes = 6
    # 关系类别: solid, non-solid (2类) 
    args.num_rel_classes = 2

    # 构建模型
    print("🔧 构建RT-DETRv3 RelTR模型...")
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    # 打印模型信息
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'📊 可训练参数数量: {n_parameters:,}')

    # 构建优化器
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() 
                   if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                      if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 构建数据集
    print("📦 构建PID数据集...")
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                               drop_last=False, collate_fn=utils.collate_fn, 
                               num_workers=args.num_workers)

    # 打印数据集信息
    print(f"📊 数据集统计:")
    print(f"   - 训练样本: {len(dataset_train)}")
    print(f"   - 验证样本: {len(dataset_val)}")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 训练批次数: {len(data_loader_train)}")

    # COCO API用于评估
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

    # 仅评估模式
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # 训练循环
    print("🚀 开始训练...")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # 训练一个epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        # 定期评估
        if epoch % args.eval_every == 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args
            )
        else:
            test_stats = {}

        # 保存检查点
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            
            # 保存最佳模型
            current_loss = train_stats.get('loss', float('inf'))
            if current_loss < best_loss:
                best_loss = current_loss
                checkpoint_paths.append(output_dir / 'best_model.pth')
            
            # 定期保存
            if epoch % args.save_every == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_epoch_{epoch}.pth')

            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 记录统计信息
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # 打印进度
        epoch_time = time.time() - epoch_start_time
        print(f'📈 Epoch {epoch}/{args.epochs-1} 完成 - 用时: {epoch_time:.1f}s')
        print(f'   训练损失: {train_stats.get("loss", "N/A"):.4f}')
        if test_stats and "coco_eval_bbox" in test_stats:
            coco_eval_bbox = test_stats.get("coco_eval_bbox", [])
            if len(coco_eval_bbox) > 0:
                print(f'   验证mAP: {coco_eval_bbox[0]:.4f}')
            else:
                print(f'   验证mAP: 暂无有效数据')
        else:
            print(f'   验证状态: 评估中...')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'🎉 训练完成! 总用时: {total_time_str}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
