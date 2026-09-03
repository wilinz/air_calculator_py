"""
图像编码器 — DeiT-Small patch embedding

将渲染的笔画灰度图 (B, 1, IMG_H, IMG_W) 编码为 64 个 patch token 序列。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from stroke_renderer import IMG_H, IMG_W


class ImageEncoder(nn.Module):
    """
    将渲染的笔画灰度图 (B, 1, IMG_H, IMG_W) 编码为 token 序列。

    使用 DeiT-Small（ImageNet 预训练）作为 backbone：
      - patch_size=16，64×256 → 4×16 = 64 个 patch tokens
      - d_model=384，与主模型维度对齐
      - 替换 patch_embed 输入通道 3→1（灰度图）
      - 位置编码从 14×14 双线性插值到 4×16
      - 冻结前 freeze_blocks 个 transformer block
    """

    def __init__(self, d_model: int = 384,
                 img_h: int = IMG_H, img_w: int = IMG_W,
                 freeze_blocks: int = 8):
        super().__init__()
        self.d_model = d_model
        patch_size = 16
        n_patches = (img_h // patch_size) * (img_w // patch_size)   # 4×16 = 64

        # 加载预训练 DeiT-Small (384 dim, 12 blocks, 6 heads)
        vit = timm.create_model('deit_small_patch16_224', pretrained=False)

        # 替换 patch embedding：3ch → 1ch
        old_proj = vit.patch_embed.proj
        new_proj = nn.Conv2d(
            1, old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        self.patch_proj = new_proj

        # 位置编码：从预训练的 14×14 插值到 4×16
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, 384))
        with torch.no_grad():
            old_pos = vit.pos_embed[:, 1:]                        # (1, 196, 384) 去掉 cls
            old_pos = old_pos.reshape(1, 14, 14, 384).permute(0, 3, 1, 2)
            new_pos = F.interpolate(old_pos,
                                    size=(img_h // patch_size, img_w // patch_size),
                                    mode='bilinear', align_corners=False)
            self.pos_embed.copy_(
                new_pos.permute(0, 2, 3, 1).reshape(1, n_patches, 384))

        # 拆分 transformer blocks：冻结前 N 个，微调后面的
        all_blocks = list(vit.blocks)
        self.blocks_frozen = nn.Sequential(*all_blocks[:freeze_blocks])
        self.blocks_trainable = nn.Sequential(*all_blocks[freeze_blocks:])
        self.norm = vit.norm

        for p in self.blocks_frozen.parameters():
            p.requires_grad_(False)

        # 如果 d_model != 384，加投影层
        self.proj = nn.Linear(384, d_model) if d_model != 384 else nn.Identity()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 1, H, W)
        Returns:
            (B, N_tok, d_model)
        """
        x = self.patch_proj(img)                                  # (B, 384, h/16, w/16)
        B, C, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, h * w, C)           # (B, 64, 384)
        x = x + self.pos_embed[:, :h * w]

        with torch.no_grad():
            x = self.blocks_frozen(x)
        x = self.blocks_trainable(x)
        x = self.norm(x)
        return self.proj(x)                                        # (B, 64, d_model)
