# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        try:
            num_patches = model.patch_embed.num_patches
        except AttributeError as err:
            num_patches = model.patch_embed[0].num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import torch.nn.functional as F


# from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid_torch


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, mlp_dim = 1600,global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, kwargs['embed_dim'] - 384))
        self.mlp = nn.Sequential(
            nn.Linear(1000, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.GELU()
        )
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward(self, x, timestamps):
        x = self.forward_features(x, timestamps)
        print("XOT",x.shape)
        x = self.head(x)
        print("XOTT",x.shape)
        # Apply MLP layer
        x = self.mlp(x)

        # Reshape to [batch_size, 40, 40]
        x = x.view(x.size(0),40, 40)
        x = x.unsqueeze(1)  # Add a channel dimension
        x = self.conv_layer(x)
        # x = conv_output.squeeze(1)  # Remove the added channel 

        return x

    def forward_features(self, x, timestamps):
        
        B = x.shape[0]
        x_list = [] 
        print(x[:, 0].shape)
        time = x.shape[1]
        for i in range(x.shape[1]) :
            x_i = self.patch_embed(x[:, i])
            # print('x_i',x_i.shape)
            x_list.append(x_i)
        

        # Concatenate along the second dimension
        x = torch.cat(x_list, dim=1)

       
        print(x.shape)

       
        # print('timestamps',timestamps.shape)
        ts_embed = get_1d_sincos_pos_embed_from_grid_torch(384, timestamps.float()).float()
        # ts_embed = F.positional_embedding(timestamps, num_embeddings=128)
        
        ts_embed = ts_embed.reshape(B,int(ts_embed.shape[0]/B),-1)
         

         
        # print('ts_embed',ts_embed[:,1,:].unsqueeze(1).shape)     
        # ts_embed = ts_embed.reshape(-1, 3, ts_embed.shape[-1]).unsqueeze(2)
       
        patch_embed_outputs = [ts_embed[:,i,:].unsqueeze(1).expand(-1,int(x.shape[1]/time) ,-1) for i in range(time)]


        # Concatenate patch embedding outputs
        ts_embed = torch.cat(patch_embed_outputs, dim=1)
        # print('patch_embed_outputs',ts_embed.shape)
       
        # ts_embed = ts_embed[1,:].unsqueeze(0).expand( int(x.shape[1]/time ),-1)
        emb = torch.cat([self.pos_embed[:, :1, :], self.pos_embed[:, 1:, :].repeat(1, time, 1)], dim=1).expand(ts_embed.shape[0], -1, -1)
        # print('pos_embed_2',emb.shape)
        ts_embed = torch.cat([torch.zeros((ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device), ts_embed], dim=1)
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)
        x = x + torch.cat(
            [torch.cat([self.pos_embed[:, :1, :], self.pos_embed[:, 1:, :].repeat(1, time, 1)], dim=1).expand(ts_embed.shape[0], -1, -1),
             ts_embed], dim=-1)
        print('xx',x.shape)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        x = self.norm(x)
        # outcome = x
        
        
        outcome = x[:, 0]
        # print("XOT",outcome.shape)
        #         # Apply MLP layer
        # x = self.mlp(x)

        # # Reshape to [batch_size, 40, 40]
        # outcome = x.view(x.size(0), 40, 40)

        

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(img_size=40,in_chans=10,
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# from models_vit import vit_large_patch16
# vit_large_patch16_nontemp = vit_large_patch16

import torch
# from your_module import vit_large_patch16  # Replace 'your_module' with the actual module containing the vit_large_patch16 function

# Assuming you have a sample input with shape (batch_size, num_channels, height, width)
# Here, I'm creating a dummy tensor with batch size 2, 3 channels, height 224, and width 224
sample_input = torch.randn((2,4 ,10, 40, 40))

# Additionally, you need to provide timestamps for each sample in the batch
# Assuming you have timestamps in the format (batch_size, num_timestamps)
# Here, I'm creating a dummy tensor with batch size 2 and 3 timestamps per sample
timestamps = torch.tensor([[0.1, 0.2, 0.3,0.35], [0.4, 0.5, 0.6,0.65]])

# Instantiate the VisionTransformer model
model = vit_large_patch16()

# Perform a forward pass
output = model(sample_input, timestamps)

# Print the output shape
print("Output Shape:", output.shape)
output = output.tolist()
croo_outptu = output[-1]
print('len',len(croo_outptu))
print(type(output))
