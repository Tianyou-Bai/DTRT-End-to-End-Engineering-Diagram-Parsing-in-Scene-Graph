import torch
import torch.nn.functional as F
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def build_dn_queries(targets, args, max_gt=200, device=None):
    """
    构造 Denoising queries (For Entity Branch)
    """
    # 获取超参
    # 如果 args 不存在这些属性，使用默认值
    num_group = getattr(args, 'dn_number', 5)
    label_noise_ratio = getattr(args, 'dn_label_noise_ratio', 0.2)
    box_noise_scale = getattr(args, 'dn_box_noise_scale', 0.4)
    num_classes = args.num_classes

    B = len(targets)
    dn_labels = []
    dn_boxes = []
    gt_num_per_img = []
    
    for t in targets:
        labels = t['labels']
        boxes = t['boxes']
        n = min(labels.shape[0], max_gt)
        gt_num_per_img.append(n)
        
        # pad labels / boxes
        pad_labels = torch.full((max_gt,), num_classes, dtype=torch.long, device=device)
        pad_boxes = torch.zeros((max_gt, 4), dtype=boxes.dtype, device=device)
        
        if n > 0:
            pad_labels[:n] = labels[:n]
            pad_boxes[:n] = boxes[:n]
            
        dn_labels.append(pad_labels)
        dn_boxes.append(pad_boxes)
    
    dn_labels = torch.stack(dn_labels, dim=0) # [B, max_gt]
    dn_boxes = torch.stack(dn_boxes, dim=0)   # [B, max_gt, 4]

    # Repeat for groups
    dn_labels = dn_labels.unsqueeze(1).repeat(1, num_group, 1) # [B, G, max_gt]
    dn_boxes = dn_boxes.unsqueeze(1).repeat(1, num_group, 1, 1) # [B, G, max_gt, 4]
    
    # Noise
    if label_noise_ratio > 0:
        noise_mask = (torch.rand_like(dn_labels.float()) < label_noise_ratio)
        rand_labels = torch.randint(0, num_classes, dn_labels.shape, device=device)
        dn_labels = torch.where(noise_mask, rand_labels, dn_labels) # noise or original
    
    if box_noise_scale > 0:
        diff = torch.zeros_like(dn_boxes)
        diff[..., :2] = dn_boxes[..., 2:] / 2
        diff[..., 2:] = dn_boxes[..., 2:]
        dn_boxes += torch.mul((torch.rand_like(dn_boxes) * 2 - 1.0), diff).cuda() * box_noise_scale
        dn_boxes = dn_boxes.clamp(0.0, 1.0)
        
    
    # Flatten groups
    dn_labels = dn_labels.view(B, -1)     # [B, G*max_gt]
    dn_boxes = dn_boxes.view(B, -1, 4)    # [B, G*max_gt, 4]
    
    dn_meta = {
        'dn_num_group': num_group,
        'dn_num_split': [dn_labels.shape[1], max_gt], # [total_dn_len, max_gt per group]
        'gt_num_per_img': gt_num_per_img,
    }
    
    return dn_labels, dn_boxes, dn_meta

def compute_dn_mask(dn_meta, num_queries, device):
    """
    Generate the attention mask [Total_Len, Total_Len]
    Total_Len = num_queries + dn_group * max_gt
    """
    num_group = dn_meta['dn_num_group']
    total_dn_len, split_gt = dn_meta['dn_num_split'] # total_dn_len = G * max_gt
    
    total_len = num_queries + total_dn_len
    
    # Initialize mask with 0 (visible)
    mask = torch.zeros((total_len, total_len), device=device)
    
    # 1. Learned queries cannot see DN queries
    mask[:num_queries, num_queries:] = float('-inf')
    
    # 2. DN queries cannot see Learned queries
    mask[num_queries:, :num_queries] = float('-inf')
    
    # 3. DN queries can only see their own group
    dn_part_mask = torch.zeros((total_dn_len, total_dn_len), device=device)
    dn_part_mask.fill_(float('-inf'))
    
    chunk_size = split_gt
    for i in range(num_group):
        start = i * chunk_size
        end = (i+1) * chunk_size
        if end > total_dn_len: end = total_dn_len
        dn_part_mask[start:end, start:end] = 0.0
        
    mask[num_queries:, num_queries:] = dn_part_mask
    
    return mask
