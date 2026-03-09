#!/usr/bin/env python3
"""PID 数据集推理 + 目标/关系 + Token Reduction 可视化脚本

功能:
1. 加载训练好的 RelTRv3 (EViT-DeiT) 模型 checkpoint
2. 对指定 PID 图像执行推理 (实体 + 关系)
3. 根据 pointer logits 还原 (subject, object) 指向
4. 过滤置信度并绘制: 边界框 + 关系连线 + 文本标签
5. 若使用 EViT 稀疏模式, 可视化被保留的 tokens (patch grid 上的点)

用法示例:
python infer_pid_visualize.py \
  --model_path output_long_run_evit_400/checkpoint0090.pth \
  --image_path data/pid_resized/123.png \
  --data_path data/pid_resized \
  --out_dir viz_results \
  --score_thresh 0.4 \
  --rel_thresh 0.4 \
  --visualize_tokens

注意:
- 依赖 matplotlib 或回退到 cv2 绘制 (首选 matplotlib)。
- 需要和训练保持相同 backbone / 结构参数 (默认当前仓库已改为 deit_tiny + EViT)。
- PID 数据集类别数量: 对象 7, 关系 2 (含一个背景在模型 softmax 末端)。
"""
import argparse
from pathlib import Path
import torch
import numpy as np
import os
import sys
# OpenCV 可能在当前环境损坏/缺失，尝试可选导入
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    cv2 = None
    _CV2_OK = False
from PIL import Image

import util.misc as utils
# Delayed import of the RelTR builder will be performed inside load_model
from datasets import build_dataset
from datasets.coco import make_coco_transforms

# 尝试使用 matplotlib (更好看的可视化)
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    USE_MPL = True
except Exception:
    USE_MPL = False


def build_infer_args():
    p = argparse.ArgumentParser("PID Inference & Visualization", add_help=True)
    p.add_argument('--model_path', required=True, type=str, help='训练好的 checkpoint 路径 (包含 model state_dict)')
    p.add_argument('--image_path', required=True, type=str, help='要推理的 PID 图像 (.png)')
    p.add_argument('--data_path', default='data/pid_resized', type=str, help='PID 数据路径 (用于拿类别与 transforms)')
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--num_entities', default=400, type=int)
    p.add_argument('--num_triplets', default=800, type=int)
    p.add_argument('--backbone', default='deit_tiny_patch16_224', type=str)
    p.add_argument('--use_evit', action='store_true', default=True)
    p.add_argument('--evit_drop_loc', default='3,5,7', type=str)
    p.add_argument('--evit_keep_rates', default='0.95,0.9,0.85', type=str)
    p.add_argument('--evit_sparse_seq', action='store_true', default=True)
    p.add_argument('--score_thresh', default=0.5, type=float, help='实体预测过滤阈值')
    p.add_argument('--rel_thresh', default=0.5, type=float, help='关系预测过滤阈值')
    p.add_argument('--out_dir', default='inference_viz', type=str)
    p.add_argument('--output_name', default=None, type=str, help='输出文件名 (默认 基于 image_path)')
    p.add_argument('--visualize_tokens', action='store_true', help='可视化 EViT 保留 tokens')
    p.add_argument('--max_rel_draw', default=50, type=int, help='最多绘制多少关系 (防爆屏)')
    p.add_argument('--dpi', default=200, type=int, help='保存图像 DPI')
    # Cleaner visualization controls
    p.add_argument('--topk_entities', default=0, type=int, help='若>0, 按得分保留前K个实体 (阈值之后, 或自动阈值前)')
    p.add_argument('--nms_thresh', default=0.0, type=float, help='若>0 启用 class-agnostic NMS (IoU>阈值抑制)')
    p.add_argument('--auto_threshold', action='store_true', help='自动上调实体阈值, 使保留数量<=topk_entities (需指定 --topk_entities>0)')
    p.add_argument('--min_score_floor', default=0.05, type=float, help='自动阈值下限 (防止过低)')
    return p


def load_model(args):
    # 构造一个最小 args 对象给 build_reltrv3 (使用训练脚本参数子集)
    class Dummy: pass
    d = Dummy()
    # 复制所需关键字段
    d.device = args.device
    d.backbone = args.backbone
    d.use_evit = args.use_evit
    d.evit_drop_loc = args.evit_drop_loc
    d.evit_keep_rates = args.evit_keep_rates
    d.evit_keep_rate = 0.9  # 备用
    d.evit_fuse_token = False
    d.evit_sparse_seq = args.evit_sparse_seq
    d.evit_sparse_grid = False
    d.evit_min_grid = 4
    d.return_interm_layers = False
    d.dilation = False
    # 位置编码 & 其它 backbone 依赖字段
    d.position_embedding = 'sine'
    d.masks = False
    d.world_size = 1
    d.distributed = False
    d.lr_backbone = 0.0  # 推理无需训练
    d.lr = 1e-4
    d.weight_decay = 1e-4

    # 模型结构相关
    d.enc_layers = 6
    d.dec_layers = 6
    d.dim_feedforward = 2048
    d.hidden_dim = 256
    d.dropout = 0.1
    d.nheads = 8
    d.num_entities = args.num_entities
    d.num_triplets = args.num_triplets
    d.pre_norm = False

    # Loss / matcher 参数 (需提供以免 build 内部访问)
    d.aux_loss = True
    d.set_cost_class = 0.5
    d.set_cost_bbox = 5.0
    d.set_cost_giou = 2.0
    d.set_iou_threshold = 0.7
    d.bbox_loss_coef = 5.0
    d.giou_loss_coef = 2.0
    d.rel_loss_coef = 1.0
    d.pointer_loss_coef = 0.25
    d.eos_coef = 0.1

    # IoU 引导参数 (保持关闭即可, 推理不需要)
    d.enable_iou_query = False
    d.iou_topk = 0
    d.enc_iou_loss_coef = 0.0

    # 任务类别 (PID)
    d.dataset_file = 'pid'
    d.num_classes = 7
    d.num_rel_classes = 2

    # 其它兼容字段
    d.use_amp = False
    d.log_gpu_memory = False

    from importlib import import_module
    build_fn = import_module('models.reltr').build_reltr
    model, criterion, post = build_fn(d)
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(args.device)
    model.eval()
    return model, post, d


def build_pid_transforms():
    # 复用 coco 验证 transforms (与训练验证一致)
    return make_coco_transforms('val')


def load_pid_metadata(data_path):
    # 直接返回 PID 训练时配置的对象/关系类别（若后续需要可改为动态读取）
    object_classes = ['cls_0','cls_1','cls_2','cls_3','cls_4','cls_5','cls_6']
    relation_classes = ['rel_0','rel_1']
    return object_classes, relation_classes


def preprocess(image_path, tfm, device):
    img = Image.open(image_path).convert('RGB')
    img_t, _ = tfm(img, None)
    return img, img_t.unsqueeze(0).to(device)


def run_inference(model, post, img_tensor, orig_img, device):
    with torch.no_grad():
        outputs = model(img_tensor)
        h, w = orig_img.size[1], orig_img.size[0]
        target_sizes = torch.tensor([[h, w]], device=device)
        results = post(outputs, target_sizes)[0]
        # 重建指针索引
        if 'sub_ptr_logits' in outputs:
            sub_idx = outputs['sub_ptr_logits'].softmax(-1).argmax(-1)[0]  # [Qr]
            obj_idx = outputs['obj_ptr_logits'].softmax(-1).argmax(-1)[0]
            results['sub_indices'] = sub_idx
            results['obj_indices'] = obj_idx
        # 附加 token reduction 元信息
        real_model = model.module if hasattr(model,'module') else model
        backbone = getattr(real_model, 'backbone', None)
        token_meta = None
        if backbone is not None and hasattr(backbone[0], 'last_token_reduction'):
            token_meta = backbone[0].last_token_reduction
        return results, token_meta


def _nms(boxes, scores, iou_thresh):
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    # boxes: [N,4] (x1,y1,x2,y2)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.sort(descending=True).indices
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remain = (iou <= iou_thresh).nonzero().squeeze(1)
        order = order[remain + 1]
    return torch.as_tensor(keep, dtype=torch.long, device=boxes.device)

def filter_results(results, score_thresh, rel_thresh, *, topk_entities=0, nms_thresh=0.0, auto_threshold=False, min_score_floor=0.05):
    scores_all = results['scores']
    # 自动阈值: 根据 topk_entities 在排序后取第k分数
    if auto_threshold and topk_entities > 0 and scores_all.numel() > topk_entities:
        sorted_scores, _ = torch.sort(scores_all, descending=True)
        th = sorted_scores[topk_entities-1].item() - 1e-6
        score_thresh = max(score_thresh, th, min_score_floor)
    ent_mask = scores_all > score_thresh
    kept_indices = ent_mask.nonzero().squeeze(1)
    # TopK 裁剪 (阈值后)
    if topk_entities > 0 and kept_indices.numel() > topk_entities:
        kept_scores = scores_all[kept_indices]
        topk_sel = kept_scores.sort(descending=True).indices[:topk_entities]
        kept_indices = kept_indices[topk_sel]
    # NMS (基于当前 kept)
    if nms_thresh > 0 and kept_indices.numel() > 0:
        nms_keep_local = _nms(results['boxes'][kept_indices], scores_all[kept_indices], nms_thresh)
        kept_indices = kept_indices[nms_keep_local]
    # 排序按分数降序 (更稳定显示)
    if kept_indices.numel() > 0:
        order_local = scores_all[kept_indices].sort(descending=True).indices
        kept_indices = kept_indices[order_local]
    # 构建实体结果
    ents = {
        'boxes': results['boxes'][kept_indices],
        'labels': results['labels'][kept_indices],
        'scores': scores_all[kept_indices],
        'orig_indices': kept_indices  # 映射到原始 query 序号
    }
    # 构建原始->新索引映射
    mapping = torch.full((scores_all.shape[0],), -1, dtype=torch.long, device=scores_all.device)
    if kept_indices.numel() > 0:
        mapping[kept_indices] = torch.arange(kept_indices.numel(), device=scores_all.device)
    # 关系过滤: 先按得分阈值, 再要求 sub/obj 都保留
    rel_scores_all = results['scores_rel']
    rel_mask_score = rel_scores_all > rel_thresh
    sub_all = results.get('sub_indices', torch.zeros_like(rel_scores_all))
    obj_all = results.get('obj_indices', torch.zeros_like(rel_scores_all))
    valid_sub = mapping[sub_all] >= 0
    valid_obj = mapping[obj_all] >= 0
    rel_mask = rel_mask_score & valid_sub & valid_obj
    if rel_mask.any():
        sub_new = mapping[sub_all[rel_mask]]
        obj_new = mapping[obj_all[rel_mask]]
        rels = {
            'sub_indices': sub_new,
            'obj_indices': obj_new,
            'rel_labels': results['labels_rel'][rel_mask],
            'rel_scores': rel_scores_all[rel_mask]
        }
    else:
        rels = {
            'sub_indices': torch.zeros((0,), dtype=torch.long),
            'obj_indices': torch.zeros((0,), dtype=torch.long),
            'rel_labels': torch.zeros((0,), dtype=torch.long),
            'rel_scores': torch.zeros((0,), dtype=rel_scores_all.dtype)
        }
    meta = {
        'applied_score_thresh': score_thresh,
        'auto_threshold': auto_threshold,
        'nms_thresh': nms_thresh,
        'topk_entities': topk_entities,
        'initial_entities': int(scores_all.numel()),
        'kept_entities': int(ents['boxes'].shape[0])
    }
    return ents, rels, meta


def visualize(orig_img, ents, rels, obj_names, rel_names, out_path, max_rel=50, token_meta=None, dpi=200):
    os.makedirs(out_path.parent, exist_ok=True)
    # 选用 matplotlib 优先
    if USE_MPL:
        fig, ax = plt.subplots(1,1, figsize=(12,8))
        ax.imshow(orig_img)
        colors = plt.cm.tab20(np.linspace(0,1,20))
        # 画实体
        for i,(box,lbl,score) in enumerate(zip(ents['boxes'], ents['labels'], ents['scores'])):
            x1,y1,x2,y2 = box.cpu().numpy()
            w,h = x2-x1, y2-y1
            col = colors[int(lbl)%20]
            ax.add_patch(Rectangle((x1,y1), w,h, edgecolor=col, facecolor='none', linewidth=2))
            name = obj_names[lbl] if int(lbl) < len(obj_names) else f'cls_{int(lbl)}'
            ax.text(x1, y1-3, f'{name}:{score:.2f}', color='white', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc=col, ec='none', alpha=0.8))
        # 画关系
        R = min(len(rels['rel_labels']), max_rel)
        for i in range(R):
            si = int(rels['sub_indices'][i]); oi = int(rels['obj_indices'][i])
            if si >= len(ents['boxes']) or oi >= len(ents['boxes']):
                continue
            sb = ents['boxes'][si].cpu().numpy(); ob = ents['boxes'][oi].cpu().numpy()
            rel_lbl = int(rels['rel_labels'][i]); rel_sc = float(rels['rel_scores'][i])
            scx = (sb[0]+sb[2])/2; scy = (sb[1]+sb[3])/2
            ocx = (ob[0]+ob[2])/2; ocy = (ob[1]+ob[3])/2
            ax.plot([scx, ocx],[scy, ocy], 'r-', linewidth=1.5, alpha=0.7)
            midx, midy = (scx+ocx)/2, (scy+ocy)/2
            rel_name = rel_names[rel_lbl] if rel_lbl < len(rel_names) else f'rel_{rel_lbl}'
            ax.text(midx, midy, f'{rel_name}:{rel_sc:.2f}', color='white', fontsize=7,
                    ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='red', ec='none', alpha=0.75))
        # Token 可视化
        if token_meta and token_meta.get('kept_idx') is not None:
            kept = token_meta['kept_idx']
            if torch.is_tensor(kept): kept = kept[0].detach().cpu()
            grid_hw = int(token_meta.get('grid_hw', 0))
            if grid_hw>0 and kept.numel()>0:
                # 近似用原图尺寸映射 (假设均匀网格)
                W,H = orig_img.size
                patch_w = W / grid_hw; patch_h = H / grid_hw
                xs = (kept % grid_hw) * patch_w + patch_w/2
                ys = (kept // grid_hw) * patch_h + patch_h/2
                ax.scatter(xs, ys, s=10, c='yellow', alpha=0.6, marker='o', edgecolors='k', linewidths=0.3, label='kept_tokens')
                ax.legend(loc='upper right', fontsize=8)
        ax.set_axis_off()
        ax.set_xlim(0, orig_img.size[0]); ax.set_ylim(orig_img.size[1],0)
        plt.title('PID Detection + Relations + Tokens', fontsize=14)
        fig.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f'[保存] {out_path}')
        plt.close(fig)
    elif _CV2_OK:
        # OpenCV 简易版
        img = np.array(orig_img).copy()
        for box,lbl,score in zip(ents['boxes'], ents['labels'], ents['scores']):
            x1,y1,x2,y2 = map(int, box.cpu().numpy())
            name = obj_names[int(lbl)] if int(lbl)<len(obj_names) else f'cls_{int(lbl)}'
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f'{name}:{score:.2f}',(x1,max(0,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        R = min(len(rels['rel_labels']), max_rel)
        for i in range(R):
            si = int(rels['sub_indices'][i]); oi = int(rels['obj_indices'][i])
            if si >= len(ents['boxes']) or oi >= len(ents['boxes']):
                continue
            sb = ents['boxes'][si].cpu().numpy(); ob = ents['boxes'][oi].cpu().numpy()
            scx = int((sb[0]+sb[2])/2); scy=int((sb[1]+sb[3])/2)
            ocx = int((ob[0]+ob[2])/2); ocy=int((ob[1]+ob[3])/2)
            cv2.line(img,(scx,scy),(ocx,ocy),(0,0,255),2)
        out_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_np)
        print(f'[保存] {out_path}')
    else:
        # 没有 matplotlib 也没有正常的 cv2，退回纯文本提示
        print('[WARN] 无法可视化: matplotlib 不可用且 OpenCV 不可用/损坏.')
        print('实体(前十):')
        for i,(b,l,s) in enumerate(zip(ents['boxes'][:10], ents['labels'][:10], ents['scores'][:10])):
            print(f'  #{i}: box={b.cpu().numpy().round(1).tolist()} label={int(l)} score={float(s):.3f}')
        print('关系(前十):')
        for i in range(min(len(rels['rel_labels']),10)):
            print(f"  #{i}: sub={int(rels['sub_indices'][i])} obj={int(rels['obj_indices'][i])} rel={int(rels['rel_labels'][i])} score={float(rels['rel_scores'][i]):.3f}")


def main():
    args = build_infer_args().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device

    if not Path(args.model_path).exists():
        print(f'模型不存在: {args.model_path}'); return
    if not Path(args.image_path).exists():
        print(f'图像不存在: {args.image_path}'); return

    print('[1] 加载模型...')
    model, post, model_cfg = load_model(args)
    print('[2] 加载类别元数据...')
    obj_names, rel_names = load_pid_metadata(args.data_path)
    print(f'对象类别: {obj_names}')
    print(f'关系类别: {rel_names}')

    print('[3] 预处理图像...')
    tfm = build_pid_transforms()
    orig_img, img_tensor = preprocess(args.image_path, tfm, device)

    print('[4] 推理...')
    results, token_meta = run_inference(model, post, img_tensor, orig_img, device)

    print('[5] 过滤结果...')
    ents, rels, meta = filter_results(
        results,
        args.score_thresh,
        args.rel_thresh,
        topk_entities=args.topk_entities,
        nms_thresh=args.nms_thresh,
        auto_threshold=args.auto_threshold,
        min_score_floor=args.min_score_floor
    )
    print(f'保留实体 {len(ents["boxes"])} / 原始 {len(results["boxes"])} | 保留关系 {len(rels["rel_labels"])} / 原始 {len(results["labels_rel"])})')
    print(f"筛选配置: score_thresh={meta['applied_score_thresh']:.4f} auto={meta['auto_threshold']} topk={meta['topk_entities']} nms={meta['nms_thresh']}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.output_name or (Path(args.image_path).stem + '_viz.png')
    out_path = out_dir / out_name

    print('[6] 可视化...')
    visualize(orig_img, ents, rels, obj_names, rel_names, out_path, max_rel=args.max_rel_draw,
              token_meta=token_meta if args.visualize_tokens else None, dpi=args.dpi)

    # 打印关系详情
    if len(rels['rel_labels'])>0:
        print('\n关系详情 (前若干):')
        for i in range(min(len(rels['rel_labels']), 20)):
            si = int(rels['sub_indices'][i]); oi = int(rels['obj_indices'][i])
            if si >= len(ents['boxes']) or oi >= len(ents['boxes']):
                continue
            sl = int(ents['labels'][si]); ol = int(ents['labels'][oi])
            rl = int(rels['rel_labels'][i]); rs = float(rels['rel_scores'][i])
            sname = obj_names[sl] if sl < len(obj_names) else f'cls_{sl}'
            oname = obj_names[ol] if ol < len(obj_names) else f'cls_{ol}'
            rname = rel_names[rl] if rl < len(rel_names) else f'rel_{rl}'
            print(f'  #{i+1}: ({sname}) -[{rname}:{rs:.2f}]-> ({oname})')

    if token_meta and args.visualize_tokens:
        kept = token_meta.get('kept_idx', None)
        if torch.is_tensor(kept):
            kept = kept[0]
        print(f'Token Reduction: 模式={token_meta.get("mode")} kept={kept.shape[0] if kept is not None else None}/{token_meta.get("total_patches")} keep_rates={token_meta.get("keep_rates")} drop_locs={token_meta.get("drop_locs")}')

    print('\n完成.')

if __name__ == '__main__':
    main()
