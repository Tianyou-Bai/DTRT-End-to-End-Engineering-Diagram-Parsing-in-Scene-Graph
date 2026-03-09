# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.pid_datasets import create_datasets

# python main.py --resume ckpt/checkpoint0149.pth --world_size 1 --output_dir ./output --epochs 170
# python main.py --world_size 1 --output_dir ./output

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
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
    parser.add_argument('--num_entities', default=1000, type=int,
                        help="Number of entity queries")
    parser.add_argument('--num_triplets', default=2000, type=int,
                        help="Number of triplet queries")
    parser.add_argument('--pre_norm', action='store_true')

    # Detection/Relation classes (set sensible defaults for PID dataset)
    parser.add_argument('--num_classes', default=12, type=int,
                        help='Number of object classes (excluding background)')
    parser.add_argument('--num_rel_classes', default=2, type=int,
                        help='Number of relation classes (excluding background)')

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
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--dataset_file', default='pid', type=str,
                        help='dataset file tag used by engine/evaluate, e.g., vg or pid')
    parser.add_argument('--ann_path', default='./data/vg/', type=str)
    parser.add_argument('--img_folder', default='data/vg/images/', type=str)

    parser.add_argument('--output_dir', default='',
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

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")

    # Architecture toggles
    parser.add_argument('--use_dinov3', action='store_true',
                        help='Use the DINOv3 RelTR architecture')

    # PID dataset options
    parser.add_argument('--pid_complete_img', action='store_true',
                        help='Use resized complete PID images instead of patched tiles')
    parser.add_argument('--pid_base_path', default='data/PID', type=str,
                        help='Base path to PID dataset root')
    parser.add_argument('--pid_max_samples', default=0, type=int,
                        help='Limit total samples for a quick run (0 disables)')

    # Lightweight RT-DETR improvements (optional)
    parser.add_argument('--enable_iou_query', action='store_true',
                        help='Enable IoU-aware query selection from encoder proposals to seed decoder queries')
    parser.add_argument('--iou_topk', default=0, type=int,
                        help='Top-K encoder proposals to pick as queries (0 means use num_entities)')

    parser.add_argument('--enable_o2m', action='store_true',
                        help='Enable one-to-many (O2M) auxiliary assignment for additional supervision')
    parser.add_argument('--o2m_topk', default=5, type=int,
                        help='Number of predictions per GT for O2M auxiliary loss')
    parser.add_argument('--o2m_iou_thresh', default=0.5, type=float,
                        help='IoU threshold to consider predictions for O2M')
    parser.add_argument('--o2m_cls_conf_thresh', default=0.0, type=float,
                        help='Optional class confidence threshold for O2M selection (0 disables)')
    # Relation matching optimization
    parser.add_argument('--rel_match_topk', default=0, type=int,
                        help='If >0, pre-filter triplet queries by a combined fg score before Hungarian relation matching')
    parser.add_argument('--rel_match_use_iou', action='store_true',
                        help='Enable IoU fusion into fg score for relation TopK pre-filter')
    parser.add_argument('--rel_match_iou_lambda_sub', default=1.0, type=float,
                        help='Weight for subject IoU in fg score fusion (only if rel_match_use_iou)')
    parser.add_argument('--rel_match_iou_lambda_obj', default=1.0, type=float,
                        help='Weight for object IoU in fg score fusion (only if rel_match_use_iou)')
    return parser


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

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # dataset_train = build_dataset(image_set='train', args=args)
    # dataset_val = build_dataset(image_set='val', args=args)
    dataset_train, dataset_val = create_datasets(
        complete_img=args.pid_complete_img or True,
        batch_size=args.batch_size,
        split_ratio=0.8,
        base_path=args.pid_base_path,
        max_samples=(args.pid_max_samples if args.pid_max_samples > 0 else None),
    )
    print("Training dataset size:", len(dataset_train))
    print("Validation dataset size:", len(dataset_val))

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
    base_ds = get_coco_api_from_dataset(dataset_val)
    # base_ds = data_loader_val.dataset.coco
    if base_ds is None:
        print("⚠️ COCO evaluator base dataset is None; COCO metrics will be skipped.")
    print("Base dataset:", base_ds)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        print(f"🔍 Analyzing checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        
        model_dict = model_without_ddp.state_dict()
        
        print(f"\n📊 Model Structure Analysis:")
        print(f"   Pretrained model has {len(pretrained_dict)} parameters")
        print(f"   Current model has {len(model_dict)} parameters")
        
        # 详细分析每一层的形状
        print(f"\n🔍 Detailed Shape Analysis:")
        print("=" * 80)
        
        compatible_count = 0
        mismatch_count = 0
        missing_count = 0
        
        # 按层分类分析
        backbone_layers = []
        transformer_layers = []
        embedding_layers = []
        classification_layers = []
        other_layers = []
        
        for name, pretrained_param in pretrained_dict.items():
            if name in model_dict:
                current_param = model_dict[name]
                status = "✅ MATCH" if pretrained_param.shape == current_param.shape else "❌ MISMATCH"
                
                # 分类存储
                layer_info = {
                    'name': name,
                    'pretrained_shape': pretrained_param.shape,
                    'current_shape': current_param.shape,
                    'status': status
                }
                
                if 'backbone' in name:
                    backbone_layers.append(layer_info)
                elif 'transformer' in name or 'encoder' in name or 'decoder' in name:
                    transformer_layers.append(layer_info)
                elif 'embed' in name:
                    embedding_layers.append(layer_info)
                elif 'class' in name or 'bbox' in name:
                    classification_layers.append(layer_info)
                else:
                    other_layers.append(layer_info)
                
                if pretrained_param.shape == current_param.shape:
                    compatible_count += 1
                else:
                    mismatch_count += 1
                    print(f"❌ {name}:")
                    print(f"     Pretrained: {pretrained_param.shape}")
                    print(f"     Current:    {current_param.shape}")
                    print(f"     Difference: {[c - p for p, c in zip(pretrained_param.shape, current_param.shape)]}")
            else:
                missing_count += 1
                print(f"❓ Missing in current model: {name}")
        
        # 显示分类统计
        print(f"\n📋 Layer Categories Analysis:")
        print(f"   🏗️  Backbone layers: {len(backbone_layers)} ({sum(1 for l in backbone_layers if l['status'] == '✅ MATCH')} compatible)")
        print(f"   🔄 Transformer layers: {len(transformer_layers)} ({sum(1 for l in transformer_layers if l['status'] == '✅ MATCH')} compatible)")
        print(f"   📎 Embedding layers: {len(embedding_layers)} ({sum(1 for l in embedding_layers if l['status'] == '✅ MATCH')} compatible)")
        print(f"   🎯 Classification layers: {len(classification_layers)} ({sum(1 for l in classification_layers if l['status'] == '✅ MATCH')} compatible)")
        print(f"   🔧 Other layers: {len(other_layers)} ({sum(1 for l in other_layers if l['status'] == '✅ MATCH')} compatible)")
        
        # 显示详细的不匹配情况
        if embedding_layers:
            print(f"\n🔍 Embedding Layers Detail:")
            for layer in embedding_layers:
                print(f"   {layer['status']} {layer['name']}: {layer['pretrained_shape']} -> {layer['current_shape']}")
        
        if classification_layers:
            print(f"\n🔍 Classification Layers Detail:")
            for layer in classification_layers:
                print(f"   {layer['status']} {layer['name']}: {layer['pretrained_shape']} -> {layer['current_shape']}")
        
        # 总结
        print(f"\n📊 Summary:")
        print(f"   ✅ Compatible layers: {compatible_count}")
        print(f"   ❌ Mismatched layers: {mismatch_count}")
        print(f"   ❓ Missing layers: {missing_count}")
        print(f"   📈 Compatibility rate: {compatible_count/(compatible_count+mismatch_count)*100:.1f}%")
        
        if mismatch_count > 0:
            print(f"\n⚠️  Found {mismatch_count} shape mismatches. Using strict=False to continue...")
            # 只加载兼容的权重
            compatible_weights = {k: v for k, v in pretrained_dict.items() 
                                if k in model_dict and v.shape == model_dict[k].shape}
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(compatible_weights, strict=False)
            print(f"   Loaded {len(compatible_weights)} compatible parameters")
            print(f"   {len(missing_keys)} parameters will be randomly initialized")
        else:
            print(f"\n✅ All layers compatible! Loading complete model...")
            model_without_ddp.load_state_dict(pretrained_dict, strict=True)
        
        # 对于形状不匹配的情况，我们只加载模型权重，不加载优化器状态
        # 这避免了广播形状不匹配的错误
        if not args.eval and 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch'] + 1
            print(f"📈 Resumed from epoch {args.start_epoch} (model weights only)")
            print("🔄 Starting with fresh optimizer/scheduler to avoid shape mismatch issues")
        elif not args.eval:
            args.start_epoch = 0
            print("🔄 Starting training from scratch")

    if args.eval:
        # 仅评估模式
        if args.resume and 'checkpoint' in locals():
            print('It is the {}th checkpoint'.format(checkpoint.get('epoch', '?')))
        else:
            print('⚠️ 未提供 --resume，将使用当前模型(可能为随机初始化)进行评估，仅用于验证评估管线。')
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)
        if args.output_dir and coco_evaluator is not None and 'bbox' in coco_evaluator.coco_eval:
            # 保存原始eval对象
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            # 另存stats为json
            try:
                stats_list = coco_evaluator.coco_eval['bbox'].stats
                if hasattr(stats_list, 'tolist'):
                    stats_list = stats_list.tolist()
                with (output_dir / 'coco_eval_bbox_stats.json').open('w') as f:
                    json.dump({'stats': stats_list}, f, indent=2)
                print('💾 已保存 COCO bbox 统计到', output_dir / 'coco_eval_bbox_stats.json')
            except Exception as e:
                print('⚠️ 保存 COCO bbox 统计失败:', e)
        return
 
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth'] # anti-crash
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
