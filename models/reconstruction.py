import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionHead(nn.Module):
    """
    Simple Convolutional Decoder to reconstruct image from feature maps.
    Used as an auxiliary task to guide token pruning.
    
    Input: Feature map (B, C, H/32, W/32)
    Output: Image (B, 3, H, W)
    """
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        
        # Swin-T last stage has 768 channels, stride 32
        # We need to upsample 5 times: 32 -> 16 -> 8 -> 4 -> 2 -> 1
        
        self.up_layers = nn.Sequential(
            # 32x -> 16x
            nn.ConvTranspose2d(in_channels, 512, kernel_size=2, stride=2),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            
            # 16x -> 8x
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            
            # 8x -> 4x
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            
            # 4x -> 2x
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            
            # 2x -> 1x
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            
            # Final projection
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor [B, C, H, W]
        Returns:
            out: Tensor [B, 3, H*32, W*32]
        """
        return self.up_layers(x)
