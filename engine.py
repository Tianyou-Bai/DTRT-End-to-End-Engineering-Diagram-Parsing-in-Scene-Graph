# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import numpy as np

import torch
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter

from datasets.coco_eval import CocoEvaluator
import util.misc as utils
from util.box_ops import rescale_bboxes
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list
from lib.openimages_evaluation import task_evaluation_sg

# 🔥 TensorBoard全局变量
tb_writer = None

def init_tensorboard(log_dir='output_optimized/tensorboard_logs'):
    """初始化TensorBoard writer"""
    global tb_writer
    if tb_writer is None:
        tb_writer = SummaryWriter(log_dir=log_dir)
        print(f"🔥 TensorBoard已初始化 - 日志目录: {log_dir}")
        print(f"🔥 启动TensorBoard: tensorboard --logdir={log_dir}")
    return tb_writer

def log_to_tensorboard(stats, epoch, phase='train'):
    """记录训练统计到TensorBoard"""
    global tb_writer
    if tb_writer is None:
        return
        
    # 记录各种loss
    if 'loss' in stats:
        tb_writer.add_scalar(f'Loss/total_{phase}', stats['loss'], epoch)
    # 统一后的目标框损失
    if 'loss_obj' in stats:
        tb_writer.add_scalar(f'Loss/obj_bbox_{phase}', stats['loss_obj'], epoch)
    if 'loss_ce' in stats:
        tb_writer.add_scalar(f'Loss/classification_{phase}', stats['loss_ce'], epoch)
    if 'loss_rel' in stats:
        tb_writer.add_scalar(f'Loss/relation_{phase}', stats['loss_rel'], epoch)
        
    # 记录错误率
    if 'class_error' in stats:
        tb_writer.add_scalar(f'Error/class_{phase}', stats['class_error'], epoch)
    if 'sub_error' in stats:
        tb_writer.add_scalar(f'Error/subject_{phase}', stats['sub_error'], epoch)
    if 'obj_error' in stats:
        tb_writer.add_scalar(f'Error/object_{phase}', stats['obj_error'], epoch)
    if 'rel_error' in stats:
        tb_writer.add_scalar(f'Error/relation_{phase}', stats['rel_error'], epoch)
        
    # 记录学习率
    if 'lr' in stats:
        tb_writer.add_scalar('Learning_Rate/main', stats['lr'], epoch)
    
    # 立即刷新到磁盘
    tb_writer.flush()
    print(f"🔥 已记录{phase}统计到TensorBoard (Epoch {epoch})")

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # 🔍 调试计数器
    batch_count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 🔍 调试：检查输入数据
        if batch_count == 0:  # 只在第一个batch打印
            print(f"🔍 Batch 0 - 样本形状: {samples.tensors.shape}")
            print(f"🔍 Batch 0 - 目标数量: {len(targets)}")
            for j, target in enumerate(targets):
                if j < 2:  # 只打印前2个目标
                    print(f"🔍 Target {j} keys: {target.keys()}")
                    if 'labels' in target:
                        print(f"🔍 Target {j} labels shape: {target['labels'].shape}")
                    if 'boxes' in target:
                        print(f"🔍 Target {j} boxes shape: {target['boxes'].shape}")
        
        batch_count += 1

        # 传入 targets 以支持 DN queries 构造
        outputs = model(samples, targets)
        # print(outputs.get('obj_boxes'))
        loss_dict = criterion(outputs, targets, inputs=samples)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # 🔍 调试：检查总loss（仅第一个batch）
        if batch_count == 1:
            print(f"🔍 总loss (加权后): {losses.item():.6f}")
            print(f"🔍 权重字典: {weight_dict}")

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        # Optional error metrics: only update if present
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'sub_error' in loss_dict_reduced:
            metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        if 'obj_error' in loss_dict_reduced:
            metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        if 'rel_error' in loss_dict_reduced:
            metric_logger.update(rel_error=loss_dict_reduced['rel_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 📊 收集统计信息并记录到TensorBoard
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_to_tensorboard(stats, epoch, 'train')

    return stats

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args, epoch=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('sub_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('obj_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('rel_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # initilize evaluator
    # TODO merge evaluation programs
    if args.dataset_file == 'vg':
        # print("Using VG dataset for evaluation")
        evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)
        if args.eval:
            evaluator_list = []
            rel_categories = getattr(data_loader.dataset, 'rel_categories', ['__background__'])
            for index, name in enumerate(rel_categories):
                if index == 0:
                    continue
                evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        else:
            evaluator_list = None
    else:
        all_results = []

    # Handle postprocessors: support dict-style {'bbox': fn} and single callable object
    if isinstance(postprocessors, dict):
        bbox_post = postprocessors.get('bbox', None)
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    else:
        bbox_post = postprocessors  # assume callable PostProcess
        iou_types = ('bbox',)

    coco_evaluator = None
    if base_ds is None:
        print("⚠️ 未提供COCO基准数据集(base_ds=None)，将跳过COCO评估阶段，仅进行关系评估与loss统计。")
    else:
        coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        # print("Samples shape:", samples.tensors.shape)
        # print("Samples:", samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        sample_device = samples.tensors.device if hasattr(samples, "tensors") else samples.device
        use_amp = bool(getattr(args, "mixed_precision", False) and sample_device.type == "cuda")
        amp_dtype = torch.float16 if getattr(args, "amp_dtype", "float16") == "float16" else torch.bfloat16
        with autocast(dtype=amp_dtype, enabled=use_amp):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # Update error metrics if they exist
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        if 'sub_error' in loss_dict_reduced:
            metric_logger.update(sub_error=loss_dict_reduced['sub_error'])
        if 'obj_error' in loss_dict_reduced:
            metric_logger.update(obj_error=loss_dict_reduced['obj_error'])
        if 'rel_error' in loss_dict_reduced:
            metric_logger.update(rel_error=loss_dict_reduced['rel_error'])

        if args.dataset_file == 'vg':
            evaluate_rel_batch(outputs, targets, evaluator, evaluator_list)
        else:
            evaluate_rel_batch_oi(outputs, targets, all_results)

        # 强制启用COCO评估器更新以获取检测性能指标
        if coco_evaluator is not None and bbox_post is not None:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            # Use the resolved bbox postprocessor
            results = bbox_post(outputs, orig_target_sizes)

            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            # print("Results:", res)
            try:
                coco_evaluator.update(res)
            except (AssertionError, KeyError) as e:
                print(f"⚠️ COCO评估器更新失败: {e}")
                print(f"   调试信息: target image_ids = {[t['image_id'].item() for t in targets]}")
                print(f"   调试信息: res keys = {list(res.keys())}")
                # 不再禁用评估器，而是继续尝试

    if args.dataset_file == 'vg':
        evaluator['sgdet'].print_stats()
    else:
        # 对于PID数据集，传递正确的关系类别信息
        # 从数据加载器获取关系类别
        pid_rel_categories = getattr(data_loader.dataset, 'rel_categories', ['solid', 'non-solid'])
        topk_value = 200  # 增加到200以覆盖更多关系
        print(f"使用 topk={topk_value} 进行PID关系评估，关系类别: {pid_rel_categories}")
        task_evaluation_sg.eval_rel_results(all_results, topk_value, do_val=True, do_vis=False, rel_categories=pid_rel_categories)

    if args.eval and args.dataset_file == 'vg':
        calculate_mR_from_evaluator_list(evaluator_list, 'sgdet')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # 强制启用COCO评估器的同步、累积和总结
    if coco_evaluator is not None:
        try:
            print("🔍 开始COCO评估器同步...")
            coco_evaluator.synchronize_between_processes()
            print("🔍 开始COCO评估器累积...")
            coco_evaluator.accumulate()
            print("🔍 开始COCO评估器总结...")
            coco_evaluator.summarize()
            print("✅ COCO评估器处理完成")
            
            # 检查评估结果是否有效
            if 'bbox' in coco_evaluator.coco_eval:
                bbox_eval = coco_evaluator.coco_eval['bbox']
                if hasattr(bbox_eval, 'stats') and bbox_eval.stats is not None:
                    print(f"📊 COCO评估统计: {bbox_eval.stats}")
                    # 友好提示：当某面积分区没有GT时，COCO会用-1.000表示N/A
                    try:
                        stats_arr = bbox_eval.stats
                        # 打印更易读的三组关键指标与面积分区说明
                        ap_all, ap50, ap75 = float(stats_arr[0]), float(stats_arr[1]), float(stats_arr[2])
                        ar_all = float(stats_arr[8]) if len(stats_arr) > 8 else None
                        def fmt(v):
                            return 'N/A' if (isinstance(v, (int, float)) and v < 0) else f"{v:.3f}"
                        print(f"📌 关键指标: mAP@[.50:.95]={fmt(ap_all)}, mAP@.50={fmt(ap50)}, mAP@.75={fmt(ap75)}, AR(all)={fmt(ar_all) if ar_all is not None else 'N/A'}")

                        # 统计当前评估集的GT在各面积分区的数量，解释为何出现-1
                        coco_gt = coco_evaluator.coco_gt
                        if coco_gt is not None:
                            img_ids = coco_gt.getImgIds()
                            ann_ids = coco_gt.getAnnIds(imgIds=img_ids)
                            anns = coco_gt.loadAnns(ann_ids)
                            s2 = 32 ** 2
                            s3 = 96 ** 2
                            small = sum(1 for a in anns if a.get('area', 0) < s2)
                            medium = sum(1 for a in anns if s2 <= a.get('area', 0) < s3)
                            large = sum(1 for a in anns if a.get('area', 0) >= s3)
                            total = len(anns)
                            print(f"🧾 GT面积分布: total={total}, small={small}, medium={medium}, large={large}")
                            if medium == 0 or large == 0 or small == 0:
                                print("ℹ️ 提示: 当某分区GT=0时，该分区的AP/AR会显示为-1.000，表示N/A，并非评估出错。")
                    except Exception as _e:
                        # 统计仅用于友好展示，失败不影响评估
                        pass
                else:
                    print("⚠️ COCO评估统计为None")
            else:
                print("⚠️ 未找到bbox评估结果")
                
        except Exception as e:
            print(f"⚠️ COCO评估器处理失败: {e}")
            print(f"   错误详情: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            # 不禁用评估器，保持为可用状态以便调试

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    # 🔥 记录验证统计到TensorBoard
    if epoch is not None:
        log_to_tensorboard(stats, epoch, 'val')
    
    if coco_evaluator is not None and bbox_post is not None:
            try:
                # 安全地处理 stats，可能是 numpy 数组或列表
                bbox_stats = coco_evaluator.coco_eval['bbox'].stats
                if bbox_stats is not None:
                    if hasattr(bbox_stats, 'tolist'):
                        stats['coco_eval_bbox'] = bbox_stats.tolist()
                    else:
                        stats['coco_eval_bbox'] = bbox_stats
                    print(f"✅ 成功获取COCO评估统计: {stats['coco_eval_bbox'][:3]}...")  # 只显示前3个值
                else:
                    print("⚠️ COCO评估统计为None，设置空列表")
                    stats['coco_eval_bbox'] = []
            except Exception as e:
                print(f"⚠️ 获取COCO评估统计失败: {e}")
                stats['coco_eval_bbox'] = []

    return stats, coco_evaluator

def evaluate_rel_batch(outputs, targets, evaluator, evaluator_list):
    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        gt_entry = {
            'gt_classes': target['labels'].cpu().clone().numpy(),
            'gt_relations': target['rel_annotations'].cpu().clone().numpy(),
            'gt_boxes': target_bboxes_scaled
        }

        # --- Pointer-based subject/object reconstruction ---
        # We removed old independent subject/object classification heads.
        # Now we derive subject/object class predictions by:
        # 1) Using pointer distributions (sub_ptr_logits/obj_ptr_logits) over entity queries to pick an entity index.
        # 2) Using the entity classification logits (pred_logits) at that entity index as the subject/object class logits.
        if 'sub_ptr_logits' in outputs and 'obj_ptr_logits' in outputs:
            entity_logits = outputs['pred_logits'][batch]                 # [Qe, C+1]
            sub_ptr = outputs['sub_ptr_logits'][batch]                    # [Qr, Qe]
            obj_ptr = outputs['obj_ptr_logits'][batch]                    # [Qr, Qe]

            sub_ptr_prob = torch.softmax(sub_ptr, dim=-1)
            obj_ptr_prob = torch.softmax(obj_ptr, dim=-1)
            sub_entity_scores, sub_entity_indices = torch.max(sub_ptr_prob, dim=-1)  # [Qr]
            obj_entity_scores, obj_entity_indices = torch.max(obj_ptr_prob, dim=-1)  # [Qr]

            # Gather entity logits for chosen indices
            sub_entity_logits = entity_logits[sub_entity_indices]         # [Qr, C+1]
            obj_entity_logits = entity_logits[obj_entity_indices]         # [Qr, C+1]
            sub_class_prob = torch.softmax(sub_entity_logits, dim=-1)
            obj_class_prob = torch.softmax(obj_entity_logits, dim=-1)
            pred_sub_scores, pred_sub_classes = torch.max(sub_class_prob[:, :-1], dim=-1)
            pred_obj_scores, pred_obj_classes = torch.max(obj_class_prob[:, :-1], dim=-1)
        else:
            # Fallback (legacy) – if old keys exist use them, else raise
            if 'sub_logits' in outputs and 'obj_logits' in outputs:
                pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
                pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
            else:
                raise KeyError("Neither pointer logits ('sub_ptr_logits','obj_ptr_logits') nor legacy ('sub_logits','obj_logits') found in model outputs.")

        # Relation logits keep the same convention: last index is background / no-relation
        rel_scores = torch.softmax(outputs['rel_logits'][batch][:, :-1], dim=-1)

        # 由于已去除独立 sub/obj bbox，这里不再提供 sub_boxes/obj_boxes，评估器需兼容
        pred_entry = {
            'sub_classes': pred_sub_classes.cpu().clone().numpy(),
            'sub_scores': pred_sub_scores.cpu().clone().numpy(),
            'obj_classes': pred_obj_classes.cpu().clone().numpy(),
            'obj_scores': pred_obj_scores.cpu().clone().numpy(),
            'rel_scores': rel_scores.cpu().clone().numpy()
        }
        # print("GT Entry:", gt_entry.get('gt_boxes'))
        # print("Pred Entry:", pred_entry)

        evaluator['sgdet'].evaluate_scene_graph_entry(gt_entry, pred_entry)

        if evaluator_list is not None:
            for pred_id, _, evaluator_rel in evaluator_list:
                gt_entry_rel = gt_entry.copy()
                mask = np.in1d(gt_entry_rel['gt_relations'][:, -1], pred_id)
                gt_entry_rel['gt_relations'] = gt_entry_rel['gt_relations'][mask, :]
                if gt_entry_rel['gt_relations'].shape[0] == 0:
                    continue
                evaluator_rel['sgdet'].evaluate_scene_graph_entry(gt_entry_rel, pred_entry)


def evaluate_rel_batch_oi(outputs, targets, all_results):

    for batch, target in enumerate(targets):
        target_bboxes_scaled = rescale_bboxes(target['boxes'].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).clone().numpy() # recovered boxes with original size

        if 'sub_ptr_logits' in outputs and 'obj_ptr_logits' in outputs:
            entity_logits = outputs['pred_logits'][batch]             # [Qe, C+1]
            entity_boxes = rescale_bboxes(outputs['pred_boxes'][batch].cpu(), torch.flip(target['orig_size'],dims=[0]).cpu()).cpu().numpy()  # [Qe,4] entity boxes in original size
            sub_ptr = outputs['sub_ptr_logits'][batch]                # [Qr, Qe]
            obj_ptr = outputs['obj_ptr_logits'][batch]
            sub_ptr_prob = torch.softmax(sub_ptr, dim=-1)
            obj_ptr_prob = torch.softmax(obj_ptr, dim=-1)
            sub_entity_scores, sub_entity_indices = torch.max(sub_ptr_prob, dim=-1)
            obj_entity_scores, obj_entity_indices = torch.max(obj_ptr_prob, dim=-1)
            sub_entity_logits = entity_logits[sub_entity_indices]
            obj_entity_logits = entity_logits[obj_entity_indices]
            sub_class_prob = torch.softmax(sub_entity_logits, dim=-1)
            obj_class_prob = torch.softmax(obj_entity_logits, dim=-1)
            pred_sub_scores, pred_sub_classes = torch.max(sub_class_prob[:, :-1], dim=-1)
            pred_obj_scores, pred_obj_classes = torch.max(obj_class_prob[:, :-1], dim=-1)
            # Gather boxes for selected entities (subject/object)
            pred_sub_boxes = entity_boxes[sub_entity_indices.cpu().numpy()]  # (Qr,4)
            pred_obj_boxes = entity_boxes[obj_entity_indices.cpu().numpy()]
        else:
            if 'sub_logits' in outputs and 'obj_logits' in outputs:
                pred_sub_scores, pred_sub_classes = torch.max(outputs['sub_logits'][batch].softmax(-1)[:, :-1], dim=1)
                pred_obj_scores, pred_obj_classes = torch.max(outputs['obj_logits'][batch].softmax(-1)[:, :-1], dim=1)
                pred_sub_boxes = np.zeros((pred_sub_scores.shape[0],4), dtype=np.float32)
                pred_obj_boxes = np.zeros((pred_obj_scores.shape[0],4), dtype=np.float32)
            else:
                raise KeyError("Neither pointer logits nor legacy subject/object logits present in outputs for OpenImages evaluation.")

        rel_scores = torch.softmax(outputs['rel_logits'][batch][:, :-1], dim=-1)

        # 对齐长度：实体查询数与关系查询数可能不同，截取共同最小长度
        num_candidates = min(
            pred_sub_scores.shape[0],
            pred_obj_scores.shape[0],
            rel_scores.shape[0]
        )
        if num_candidates <= 0:
            continue
        pred_sub_scores = pred_sub_scores[:num_candidates]
        pred_sub_classes = pred_sub_classes[:num_candidates]
        pred_obj_scores = pred_obj_scores[:num_candidates]
        pred_obj_classes = pred_obj_classes[:num_candidates]
        rel_scores = rel_scores[:num_candidates]

        relation_idx = target['rel_annotations'].cpu().numpy()
        gt_sub_boxes = target_bboxes_scaled[relation_idx[:, 0]]
        gt_sub_labels = target['labels'][relation_idx[:, 0]].cpu().clone().numpy()
        gt_obj_boxes = target_bboxes_scaled[relation_idx[:, 1]]
        gt_obj_labels = target['labels'][relation_idx[:, 1]].cpu().clone().numpy()

        img_result_dict = {
            'sbj_labels': pred_sub_classes.cpu().clone().numpy(),
            'sbj_scores': pred_sub_scores.cpu().clone().numpy(),
            'obj_labels': pred_obj_classes.cpu().clone().numpy(),
            'obj_scores': pred_obj_scores.cpu().clone().numpy(),
            'prd_scores': rel_scores.cpu().clone().numpy(),
            'sbj_boxes': pred_sub_boxes,
            'obj_boxes': pred_obj_boxes,
            'image': str(target['image_id'].item()) + '.jpg',
            'gt_sbj_boxes': gt_sub_boxes,
            'gt_sbj_labels': gt_sub_labels,
            'gt_obj_boxes': gt_obj_boxes,
            'gt_obj_labels': gt_obj_labels,
            'gt_prd_labels': relation_idx[:, 2]
        }
        all_results.append(img_result_dict)
