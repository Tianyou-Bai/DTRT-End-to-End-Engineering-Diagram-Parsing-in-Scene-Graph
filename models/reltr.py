# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .dn_utils import build_dn_queries, compute_dn_mask
from .reconstruction import ReconstructionHead

class RelTR(nn.Module):
    """ RelTR: Relation Transformer for Scene Graph Generation """
    def __init__(self, backbone, transformer, num_classes, num_rel_classes, num_entities, num_triplets, aux_loss=False, matcher=None, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of entity classes
            num_entities: number of entity queries
            num_triplets: number of coupled subject/object queries
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_entities = num_entities
        self.transformer = transformer
        self.args = args # Save args for DN config
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        
        # Reconstruction Head (Optional)
        self.enable_reconstruction = getattr(args, 'enable_reconstruction', False)
        if self.enable_reconstruction:
            print("🧱 Reconstruction Head Enabled")
            # Backbone num_channels is usually last layer channels
            # Swin-T last layer has 768 channels
            self.recon_head = ReconstructionHead(in_channels=backbone.num_channels)

        self.entity_embed = nn.Embedding(num_entities, hidden_dim*2)
        self.triplet_embed = nn.Embedding(num_triplets, hidden_dim*3)
        self.so_embed = nn.Embedding(2, hidden_dim) # subject and object encoding

        # entity prediction
        self.entity_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.entity_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        
        # DN: Extra embedding for label encoding and box 2 embedding
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim)
        self.ref_point_head = MLP(4, hidden_dim, hidden_dim, 2) # Maps (x,y,w,h) to pos embed

        # mask head
        self.so_mask_conv = nn.Sequential(torch.nn.Upsample(size=(28, 28)),
                                          nn.Conv2d(2, 64, kernel_size=3, stride=2, padding=3, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(64),
                                          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.BatchNorm2d(32))
        self.so_mask_fc = nn.Sequential(nn.Linear(2048, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 128))

        # predicate classification
        self.rel_class_embed = MLP(hidden_dim*2+128, hidden_dim, num_rel_classes + 1, 2)

        # subject/object label classfication and box regression
        self.sub_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, samples: NestedTensor, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the entity classification logits (including no-object) for all entity queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": the normalized entity boxes coordinates for all entity queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "sub_logits": the subject classification logits
               - "obj_logits": the object classification logits
               - "sub_boxes": the normalized subject boxes coordinates
               - "obj_boxes": the normalized object boxes coordinates
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        backbone_out = self.backbone(samples)
        if len(backbone_out) == 4:
            features, pos, decisions, scores = backbone_out
        else:
            features, pos = backbone_out
            decisions, scores = None, None
        
        src, mask = features[-1].decompose()
        assert mask is not None
        
        # --- CDN (Contrastive DeNoising) Logic ---
        input_entity_embed = self.entity_embed.weight
        dn_meta = None
        tgt_mask = None
        triplet_entity_key_padding_mask = None
        
        if self.training and targets is not None and self.args is not None:
             # Generate DN queries
             dn_labels, dn_boxes, dn_meta = build_dn_queries(targets, self.args, max_gt=200, device=src.device)
             
             # Create embeddings
             # Content: Label embedding
             dn_content = self.label_enc(dn_labels)
             # Pos: Box -> MLP -> Embedding
             dn_pos = self.ref_point_head(dn_boxes)
             
             # Concat content and pos to match RelTR structure of [pos, content] separation or concat?
             # RelTR transformer splits input 'entity_embed' (d*2) into pos (d) and content (d).
             # So we must concat dn_pos and dn_content along dim -1
             dn_embed = torch.cat([dn_pos, dn_content], dim=-1) # [B, N_dn, 2*D]
             
             # Expand learned queries for batch
             bs = src.shape[0]
             dataset_embed = input_entity_embed.unsqueeze(0).expand(bs, -1, -1)
             
             # Concat all queries: [B, N_all, 2*D]
             # Note: learned queries come first, DN queries second
             input_entity_embed = torch.cat([dataset_embed, dn_embed], dim=1)
             
             # Generate Mask for Self-Attention
             tgt_mask = compute_dn_mask(dn_meta, self.num_entities, src.device)
             
             # Generate Mask for Triplet Cross-Attention (Hide DN queries)
             # Shape [B, N_all]
             # First num_entities = False (Keep), Rest = True (Ignore)
             # mask True indicates "padding" (ignored)
             dn_len = dn_embed.shape[1]
             triplet_entity_key_padding_mask = torch.zeros((bs, self.num_entities + dn_len), dtype=torch.bool, device=src.device)
             triplet_entity_key_padding_mask[:, self.num_entities:] = True

        # ----------------------------------------

        hs, hs_t, so_masks, _ = self.transformer(self.input_proj(src), mask, input_entity_embed,
                                                 self.triplet_embed.weight, pos[-1], self.so_embed.weight,
                                                 tgt_mask=tgt_mask,
                                                 triplet_entity_key_padding_mask=triplet_entity_key_padding_mask) 
        
        # Split outputs if DN was used
        if dn_meta is not None:
            # hs shape: [Layer, B, N_all, D]
            dn_len = dn_meta['dn_num_split'][0]
            # First Self.num_entities are the real outputs
            hs_dn = hs[:, :, self.num_entities:, :]
            hs = hs[:, :, :self.num_entities, :]
            
            # We don't use DN outputs for relation currently, just entities
            # Relation branch relies on 'hs_t' (triplets) which we didn't add DN to.
        
        # [6, 16, 100, 256], [6, 16, 200, 512], [6, 16, 200, 2, 25, 25]
        # print(so_masks.shape)
        so_masks = so_masks.detach()
        so_masks = self.so_mask_conv(so_masks.view(-1, 2, src.shape[-2],src.shape[-1])).view(hs_t.shape[0], hs_t.shape[1], hs_t.shape[2],-1)
        so_masks = self.so_mask_fc(so_masks)

        hs_sub, hs_obj = torch.split(hs_t, self.hidden_dim, dim=-1)
        # print(hs_obj)
        outputs_class = self.entity_class_embed(hs)
        outputs_coord = self.entity_bbox_embed(hs).sigmoid()

        outputs_class_sub = self.sub_class_embed(hs_sub)
        outputs_coord_sub = self.sub_bbox_embed(hs_sub).sigmoid()

        outputs_class_obj = self.obj_class_embed(hs_obj)
        outputs_coord_obj = self.obj_bbox_embed(hs_obj).sigmoid() # [6, 16, 200, 4]
        # print(outputs_coord_obj[-1])

        outputs_class_rel = self.rel_class_embed(torch.cat((hs_sub, hs_obj, so_masks), dim=-1))

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'sub_logits': outputs_class_sub[-1], 'sub_boxes': outputs_coord_sub[-1],
               'obj_logits': outputs_class_obj[-1], 'obj_boxes': outputs_coord_obj[-1],
               'rel_logits': outputs_class_rel[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                                                    outputs_class_obj, outputs_coord_obj, outputs_class_rel)
            
        if dn_meta is not None:
             out['dn_outputs'] = {
                 'dn_meta': dn_meta,
                 'hs_dn': hs_dn,
                 'pred_logits': self.entity_class_embed(hs_dn),
                 'pred_boxes': self.entity_bbox_embed(hs_dn).sigmoid()
             }
        
        if decisions is not None:
            out['decisions'] = decisions
            out['scores'] = scores
            
        if self.enable_reconstruction:
            # Use the last feature map (usually index '3' or '7')
            # features is a list of NestedTensors.
            # features[-1] is the deepest supervision feature.
            # For Swin, it's the 32x downsampled feature.
            feature_last = features[-1].tensors # [B, C, H, W]
            recon_img = self.recon_head(feature_last)
            
            # Since input was padded (NestedTensor), loss calculation should ideally respect mask?
            # Or we simply assume reconstruction of padded area is 0 (or irrelevant).
            # For simplicity, we return full tensor.
            out['pred_reconstruction'] = recon_img

        return out
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_class_sub, outputs_coord_sub,
                      outputs_class_obj, outputs_coord_obj, outputs_class_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'sub_logits': c, 'sub_boxes': d, 'obj_logits': e, 'obj_boxes': f,
                 'rel_logits': g}
                for a, b, c, d, e, f, g in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_sub[:-1],
                                               outputs_coord_sub[:-1], outputs_class_obj[:-1], outputs_coord_obj[:-1],
                                               outputs_class_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, num_rel_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.num_rel_classes = num_rel_classes
        empty_weight_rel = torch.ones(num_rel_classes+1)
        empty_weight_rel[-1] = self.eos_coef
        self.register_buffer('empty_weight_rel', empty_weight_rel)
        
    def compute_dn_loss(self, dn_outputs, targets):
        """
        Compute additional loss for Denoising Queries
        """
        dn_meta = dn_outputs['dn_meta']
        num_group = dn_meta['dn_num_group']
        total_dn_len, max_gt = dn_meta['dn_num_split']
        gt_num_per_img = dn_meta['gt_num_per_img']
        mapping = []
        for n in gt_num_per_img:
             # Create mapping to valid GT indices
             m = torch.arange(n).long()
             pad = torch.full((max_gt - n,), -1).long()
             mapping.append(torch.cat([m, pad]))
        
        # We need to compute loss for each group
        # The outputs are [Layer, B, N_dn, C] (logits) and [Layer, B, N_dn, 4] (boxes)
        # But here we take the last layer outputs mainly, or aux
        
        # Currently `dn_outputs` dictionary has:
        # pred_logits: [B, N_dn, num_classes+1]
        # pred_boxes: [B, N_dn, 4]
        
        # Use simple DN (last layer only for now to fix shape/batch issues)
        # dn_outputs['pred_logits'] is [Layers, B, Q, C]
        pred_logits = dn_outputs['pred_logits'][-1]
        pred_boxes = dn_outputs['pred_boxes'][-1]
        device = pred_logits.device
        
        # Prepare targets for DN
        # We need to tile the GTs to match groups
        target_labels = []
        target_boxes = []
        for i, t in enumerate(targets):
            labels = t['labels']
            boxes = t['boxes']
            n = gt_num_per_img[i]
            
            # For each group, we have 'max_gt' queries.
            # The first 'n' map to the real GTs. The rest are "no object".
            
            cur_labels = torch.full((max_gt,), self.num_classes, dtype=torch.long, device=device) # Init as NoObject
            cur_labels[:n] = labels[:n]
            
            cur_boxes = torch.zeros((max_gt, 4), dtype=boxes.dtype, device=device)
            cur_boxes[:n] = boxes[:n]
            
            target_labels.append(cur_labels.repeat(num_group))
            target_boxes.append(cur_boxes.repeat(num_group, 1))
            
        target_labels = torch.stack(target_labels) # [B, G*max_gt]
        target_boxes = torch.stack(target_boxes)   # [B, G*max_gt, 4]
        
        # Compute losses
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_labels, self.empty_weight)
        
        # Box Loss (L1 & Sigmoid)
        # Only compute box loss for valid objects (label != num_classes)
        # However, due to batching, mask is easier.
        mask = (target_labels != self.num_classes)
        
        if mask.sum() > 0:
            loss_bbox = F.l1_loss(pred_boxes[mask], target_boxes[mask], reduction='mean')
            
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(pred_boxes[mask]),
                box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))
            loss_giou = loss_giou.mean()
        else:
            loss_bbox = torch.tensor(0.0).to(device)
            loss_giou = torch.tensor(0.0).to(device)
            
        losses = {
            'dn_loss_ce': loss_ce,
            'dn_loss_bbox': loss_bbox,
            'dn_loss_giou': loss_giou
        }
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Entity/subject/object Classification loss
        """
        assert 'pred_logits' in outputs

        pred_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices[0])
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices[0])])
        target_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o

        sub_logits = outputs['sub_logits']
        obj_logits = outputs['obj_logits']

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]] for t, (_, J) in zip(targets, indices[1])])
        
        target_relo_classes_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]] for t, (_, J) in zip(targets, indices[1])])

        target_sub_classes = torch.full(sub_logits.shape[:2], self.num_classes, dtype=torch.int64, device=sub_logits.device)
        target_obj_classes = torch.full(obj_logits.shape[:2], self.num_classes, dtype=torch.int64, device=obj_logits.device)

        target_sub_classes[rel_idx] = target_rels_classes_o
        target_obj_classes[rel_idx] = target_relo_classes_o

        target_classes = torch.cat((target_classes, target_sub_classes, target_obj_classes), dim=1)
        src_logits = torch.cat((pred_logits, sub_logits, obj_logits), dim=1)

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction='none')

        loss_weight = torch.cat((torch.ones(pred_logits.shape[:2]).to(pred_logits.device), indices[2]*0.5, indices[3]*0.5), dim=-1)
        losses = {'loss_ce': (loss_ce * loss_weight).sum()/self.empty_weight[target_classes].sum()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(pred_logits[idx], target_classes_o)[0]
            losses['sub_error'] = 100 - accuracy(sub_logits[rel_idx], target_rels_classes_o)[0]
            losses['obj_error'] = 100 - accuracy(obj_logits[rel_idx], target_relo_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['rel_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["rel_annotations"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the entity/subject/object bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices[0])
        pred_boxes = outputs['pred_boxes'][idx]
        target_entry_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices[0])], dim=0)

        rel_idx = self._get_src_permutation_idx(indices[1])
        target_rels_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 0]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        target_relo_boxes = torch.cat([t['boxes'][t["rel_annotations"][i, 1]] for t, (_, i) in zip(targets, indices[1])], dim=0)
        rels_boxes = outputs['sub_boxes'][rel_idx]
        relo_boxes = outputs['obj_boxes'][rel_idx]

        src_boxes = torch.cat((pred_boxes, rels_boxes, relo_boxes), dim=0)
        target_boxes = torch.cat((target_entry_boxes, target_rels_boxes, target_relo_boxes), dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_relations(self, outputs, targets, indices, num_boxes, log=True):
        """Compute the predicate classification loss
        """
        assert 'rel_logits' in outputs

        src_logits = outputs['rel_logits']
        idx = self._get_src_permutation_idx(indices[1])
        target_classes_o = torch.cat([t["rel_annotations"][J,2] for t, (_, J) in zip(targets, indices[1])])
        target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_rel)

        losses = {'loss_rel': loss_ce}
        if log:
            losses['rel_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'relations': self.loss_relations
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, inputs=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             inputs: (optional) Expected keys: tensor, mask 
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        self.indices = indices

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"])+len(t["rel_annotations"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels' or loss == 'relations':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        # Compute DN Loss
        if 'dn_outputs' in outputs:
            dn_loss = self.compute_dn_loss(outputs['dn_outputs'], targets)
            losses.update(dn_loss)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def build_reltr(args):
    """Build original RelTR model (compatibility builder).

    This mirrors other builders (e.g. build_reltr_dinov3) so scripts that
    expect a build_* function can use the original RelTR implementation.
    """
    device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = RelTR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_rel_classes=args.num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=getattr(args, 'aux_loss', False),
        args=args
    )

    matcher = build_matcher(args)
    model.matcher = matcher
    model.args = args

    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': getattr(args, 'bbox_loss_coef', 5.0),
        'loss_giou': getattr(args, 'giou_loss_coef', 2.0),
        'loss_rel': getattr(args, 'rel_loss_coef', 1.0),
    }
    
    # Add DN loss weights
    weight_dict['dn_loss_ce'] = 1.0
    weight_dict['dn_loss_bbox'] = getattr(args, 'bbox_loss_coef', 5.0)
    weight_dict['dn_loss_giou'] = getattr(args, 'giou_loss_coef', 2.0)

    if getattr(args, 'aux_loss', False):
        aux_weight_dict = {}
        for i in range(getattr(args, 'dec_layers', getattr(args, 'dec_layers', 6)) - 1):
            for k, v in weight_dict.items():
                aux_weight_dict[f"{k}_{i}"] = v
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'relations', 'cardinality']

    criterion = SetCriterion(
        args.num_classes,
        args.num_rel_classes,
        matcher,
        weight_dict,
        getattr(args, 'eos_coef', 0.1),
        losses=losses,
    )
    
    if 'dynamic' in args.backbone:
        from .dynamic_loss import DistillDiffPruningLoss
        # Teacher model can be passed here if available
        teacher_model = None 
        # Note: To enable feature distillation, load a pre-trained teacher model here
        
        criterion = DistillDiffPruningLoss(
            teacher_model=teacher_model,
            base_criterion=criterion,
            ratio_weight=getattr(args, 'ratio_weight', 2.0),
            distill_weight=getattr(args, 'distill_weight', 0.5),
            pruning_loc=getattr(args, 'pruning_loc', [1, 1, 5]),
            keep_ratio=getattr(args, 'sparse_ratio', [0.5, 0.4, 0.3]),
            reconstruction_loss_coef=getattr(args, 'recon_loss_coef', 1.0)
        )
        # Update weight_dict for engine to pick up sparsity loss
        weight_dict['loss_sparsity'] = getattr(args, 'ratio_weight', 2.0)
        
        if getattr(args, 'enable_reconstruction', False):
             weight_dict['loss_reconstruction'] = getattr(args, 'recon_loss_coef', 1.0)


    criterion.weight_dict = weight_dict
    criterion.args = args
    criterion.to(device)

    postprocessors = PostProcess()

    model.to(device)

    return model, criterion, postprocessors


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):

    if hasattr(args, 'dataset_file') and args.dataset_file == 'pid':
        num_classes = 13  # PID数据集的对象类别数量（12类+1个no_object）
        num_rel_classes = 3  # PID数据集的关系类别数量（2类+1个no_relation）
    elif hasattr(args, 'dataset') and args.dataset == 'pid':
        num_classes = 13  # PID数据集的对象类别数量（12类+1个no_object）
        num_rel_classes = 3  # PID数据集的关系类别数量（2类+1个no_relation）
    else:
        num_classes = 151 if (hasattr(args, 'dataset') and args.dataset != 'oi') else 289 # some entity categories in OIV6 are deactivated.
        num_rel_classes = 51 if (hasattr(args, 'dataset') and args.dataset != 'oi') else 31

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    matcher = build_matcher(args)
    model = RelTR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_rel_classes = num_rel_classes,
        num_entities=args.num_entities,
        num_triplets=args.num_triplets,
        aux_loss=args.aux_loss,
        matcher=matcher)

    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_rel'] = args.rel_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', "relations"]

    criterion = SetCriterion(num_classes, num_rel_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors

