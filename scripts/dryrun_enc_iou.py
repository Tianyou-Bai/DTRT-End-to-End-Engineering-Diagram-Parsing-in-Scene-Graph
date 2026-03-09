import torch
from models.reltrv3 import build_reltrv3

class Args: pass
args=Args()
args.device='cpu'
args.num_classes=7
args.num_rel_classes=2
args.num_entities=50
args.num_triplets=100
args.aux_loss=False
args.bbox_loss_coef=5.0
args.giou_loss_coef=2.0
args.rel_loss_coef=1.0
args.pointer_loss_coef=0.25
args.eos_coef=0.1
args.enable_iou_query=True
args.enc_iou_loss_coef=0.2
args.hidden_dim=128
args.nheads=8
args.enc_layers=1
args.dec_layers=1
args.dim_feedforward=256
args.dropout=0.1
args.num_feature_levels=4
args.dec_n_points=4
args.enc_n_points=4
args.backbone='resnet50'
args.return_interm_layers=True
args.enc_iou_pos_thresh=0.5
args.enc_iou_max_samples=32
args.enc_iou_debug=True
args.position_embedding='sine'
args.masks=False
args.dilation=False
args.lr=1e-4
args.lr_backbone=1e-5
args.lr_drop=100
args.world_size=1
args.distributed=False
args.gpu=0
args.set_cost_class=1.0
args.set_cost_bbox=5.0
args.set_cost_giou=2.0
args.set_iou_threshold=0.7

model, criterion, _ = build_reltrv3(args)
criterion.enc_iou_pos_thresh=args.enc_iou_pos_thresh
criterion.enc_iou_max_samples=args.enc_iou_max_samples
criterion.enc_iou_debug=True
criterion._enc_iou_debug_calls=0

from util.misc import nested_tensor_from_tensor_list
x=torch.randn(2,3,224,224)
out=model(nested_tensor_from_tensor_list([x[0],x[1]]))
print('Output has enc_scores:', 'enc_scores' in out)
print('enc_scores shape:', out.get('enc_scores', None).shape if 'enc_scores' in out else None)
if 'enc_scores' in out:
	print('enc_scores sigmoid mean:', out['enc_scores'].sigmoid().mean().item())
T=[]
for b in range(2):
	num_obj=4
	labels=torch.randint(0,7,(num_obj,))
	boxes=torch.rand(num_obj,4)
	# 构造2个关系: [sub_idx, rel_label(1), obj_idx]
	rels=[]
	if num_obj>=2:
		rels.append([0,1,1])
		rels.append([1,1,2 if num_obj>2 else 0])
	rel_annotations=torch.tensor(rels,dtype=torch.long)
	T.append({'labels':labels,'boxes':boxes,'rel_annotations':rel_annotations})
losses=criterion(out,T)
print('criterion.losses:', criterion.losses)
print('criterion.weight_dict keys:', criterion.weight_dict.keys())
print('Returned loss keys:', list(losses.keys()))
print('loss_enc_iou value:', float(losses.get('loss_enc_iou', torch.tensor(-1.0))))
