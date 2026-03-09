
import torch
import torch.nn.functional as F
import torch.nn as nn

class DistillDiffPruningLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    Adapted for RelTR.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, pruning_loc=[1, 1, 5], keep_ratio=[0.5, 0.4, 0.3], clf_weight=0, mse_token=False, print_mode=True, reconstruction_loss_coef=1.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.reconstruction_loss_coef = reconstruction_loss_coef

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight, 'reconstruction_weight', reconstruction_loss_coef)

    def forward(self, outputs, targets, inputs=None):
        # outputs is the student output dict
        # inputs is the NestedTensor usually
        
        # 1. Base DETR/RelTR Loss
        cls_loss = self.base_criterion(outputs, targets, inputs)
        
        new_losses = cls_loss.copy()
        
        # 2. Extract Dynamic Outputs
        decisions = outputs.get('decisions', None)
        scores = outputs.get('scores', None)
        
        if decisions is not None and scores is not None:
             # Calculate Sparsity Loss...
             pred_loss = 0.0
             ratio = self.keep_ratio
             for i, score in enumerate(scores):
                 probs = torch.exp(score)
                 p_keep = probs[:, :, 0]
                 pos_ratio = p_keep.mean(dim=1)
                 pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()
             
             new_losses['loss_sparsity'] = pred_loss

        # 3. Reconstruction Loss (Self-Supervised for Scorer)
        pred_recon = outputs.get('pred_reconstruction', None)
        if pred_recon is not None and inputs is not None:
            # inputs.tensors is [B, 3, H, W]
            input_imgs = inputs.tensors
            
            # Ensure sizes match (handling padding if necessary)
            # Resize pred to input if slightly different due to padding requirements of backbone?
            if pred_recon.shape != input_imgs.shape:
                pred_recon = F.interpolate(pred_recon, size=input_imgs.shape[2:], mode='bilinear', align_corners=False)
                
            # If using NestedTensor, we might want to mask the loss?
            # inputs.mask exists: 1 on padded pixels.
            # We want to ignore padded pixels.
            if inputs.mask is not None:
                # Expand mask to [B, 3, H, W]
                # mask is [B, H, W]
                mask = inputs.mask.unsqueeze(1).float()
                # valid pixels are 0 in mask? 
                # RelTR NestedTensor: "containing 1 on padded pixels"
                valid_mask = (1 - mask)
                
                loss_recon = F.mse_loss(pred_recon * valid_mask, input_imgs * valid_mask, reduction='sum') / (valid_mask.sum() + 1e-6)
            else:
                loss_recon = F.mse_loss(pred_recon, input_imgs)
                
            new_losses['loss_reconstruction'] = loss_recon

        # 4. Distillation (if teacher exists)
        # ... (Teacher logic)
        
        return new_losses
