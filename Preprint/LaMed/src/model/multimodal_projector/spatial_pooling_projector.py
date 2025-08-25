from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.init as init

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

import copy
import os
import math
import numpy as np
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class resolution_attention_v3(nn.Module):

    def __init__(self, in_channels=16, out_channels=8, emb_dim=768, output_dim=768, dropout=0.1, aropout=0.0):
        super(resolution_attention_v3, self).__init__()
        self.emb_dim = emb_dim
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.attn = None
        self.output_linear = nn.Linear(emb_dim,emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(emb_dim) 
    
    def forward(self, LR_image, HR_image, kernel_size, context=None, mask=None):
        S_d, S_w, S_h = kernel_size[0], kernel_size[1], kernel_size[2] 
        spatial_d, spatial_w, spatial_h = HR_image.size(1), HR_image.size(2), HR_image.size(3)

        batch = LR_image.size(0)
        dimention = LR_image.size(-1)
        LR_image_input = LR_image.contiguous().view(batch, -1, dimention).unsqueeze(1).permute(0, 2, 1, 3)
        
        HR_image_input = HR_image.reshape(batch, (spatial_d//S_d)*S_d, 1, (spatial_w//S_w)*S_w, 1, (spatial_h//S_h)*S_h, 1, dimention).view(batch, (spatial_d//S_d), S_d, (spatial_w//S_w), S_w, (spatial_h//S_h), S_h, dimention).permute(0, 2, 4, 6, 1, 3, 5, 7) 
        HR_image_input = HR_image_input.contiguous().view(batch, S_d*S_w*S_h, (spatial_d//S_d)*(spatial_w//S_w)*(spatial_h//S_h), dimention).permute(0, 2, 1, 3)  

        query_list=self.Wq(LR_image_input)
        key_list=self.Wk(HR_image_input) 
        value_list=self.Wv(HR_image_input) 
        x, self.attn = attention(query_list, key_list, value_list, mask=mask, dropout=self.dropout) 

        x = x.view(batch, -1, dimention)
        query_list = query_list.view(batch, -1, dimention)
        
        x = self.output_linear(x) 
        x = self.norm(query_list + self.dropout_2(x)) 
        return x

class VisualPacker_3d(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim  
        self.out_dim = out_dim 
        
        self.proj_mpls = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.kernel_size = (2, 2, 2)
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)] 
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))]
        
        self.resolution_attention = resolution_attention()

    def forward(self, visual_inputs):
        batch = visual_inputs.size(0)
        HR_features = visual_inputs.view(batch, 8, 16, 16, 768) 
        LR_features = F.avg_pool3d(HR_features.clone().permute(0, 4, 1, 2, 3), kernel_size=self.kernel_size).permute(0, 2, 3, 4, 1)

        absorbed_points = self.resolution_attention(LR_features, HR_features) 

        feats = absorbed_points
        visual_embeddings = self.proj_mpls(feats).view(batch, -1, self.out_dim) 

        return visual_embeddings
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num

class VisualPacker_3d_phi_v3(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.proj_mpls = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.kernel_size = (1, 4, 4) 
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))] 
        
        self.resolution_attention = resolution_attention_v3()

    def forward(self, visual_inputs):
        batch = visual_inputs.size(0)
        HR_features = visual_inputs.view(batch, 8, 16, 16, 768)
        LR_features = F.avg_pool3d(HR_features.clone().permute(0, 4, 1, 2, 3), kernel_size=self.kernel_size).permute(0, 2, 3, 4, 1)
        absorbed_points = self.resolution_attention(LR_features, HR_features, self.kernel_size)
        feats = absorbed_points
        visual_embeddings = self.proj_mpls(feats).view(batch, -1, self.out_dim)

        return visual_embeddings
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num


class VisualPacker_3d_phi_v3_control_kernel(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='(1_4_4)', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if pooling_type == '(1_4_4)':
            self.kernel_size = (1, 4, 4)
        elif pooling_type == '(2_2_2)':
            self.kernel_size = (2, 2, 2)
        elif pooling_type == '(4_4_2)':
            self.kernel_size = (2, 4, 4)
        elif pooling_type == '(8_8_1)':
            self.kernel_size = (1, 8, 8)
        else:
            print("Invalid pooling type. Choose from '(1_4_4)', '(2_2_2)', '(4_4_2)', or '(8_8_1)'.")
            print("Using default pooling type '(1_4_4)'")
            self.kernel_size = (1, 4, 4)

        self.proj_mpls = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]  
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))] 
        
        self.resolution_attention = resolution_attention_v3()

    def forward(self, visual_inputs):
        # visual_inputs 
        batch = visual_inputs.size(0)
        # high resolution features
        HR_features = visual_inputs.view(batch, 8, 16, 16, 768) 
        # low resolution points
        LR_features = F.avg_pool3d(HR_features.clone().permute(0, 4, 1, 2, 3), kernel_size=self.kernel_size).permute(0, 2, 3, 4, 1) 
        absorbed_points = self.resolution_attention(LR_features, HR_features, self.kernel_size)
        feats = absorbed_points
        # Projection
        visual_embeddings = self.proj_mpls(feats).view(batch, -1, self.out_dim) 

        return visual_embeddings
    
    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num


class SpatialPoolingProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        
        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)] 
        self.num_patches_post = [num // pooling_size for num in self.num_patches_pre] 

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x): 
        B = x.shape[0] # B*N*D

        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x) 
            x = F.avg_pool3d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x) 
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1)
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1)

        x = rearrange(x, "b n d -> (b n) d") 
        x = self.projector(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B)  

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num


class SpatialPoolingProjector2(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.kernel_size = (1, 4, 4)

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))] 

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x):  
        B = x.shape[0] # B*N*D

        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x) 
            x = F.avg_pool3d(x, kernel_size=self.kernel_size) 
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x) 
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1) #b d n
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1) #b n d

        x = rearrange(x, "b n d -> (b n) d")  
        x = self.projector(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B) 

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num

class ablation_spatialpooling_Projector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.kernel_size = (1, 4, 4) 

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)] 
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))]

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x):
        B = x.shape[0]

        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.avg_pool3d(x, kernel_size=self.kernel_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x) 
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1)
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1) 

        x = rearrange(x, "b n d -> (b n) d")

        x = self.projector(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B) 

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num

class ablation_sequencepooling_Projector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.pooling_out_num = 128

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x): 
        B = x.shape[0] # B*N*D

        x = x.permute(0, 2, 1) #b d n
        x = F.avg_pool1d(x, kernel_size=self.pooling_size**4, stride=self.pooling_size**4)
        x = x.permute(0, 2, 1) #b n d

        x = rearrange(x, "b n d -> (b n) d") 

        x = self.projector(x) 

        x = rearrange(x, "(b n) d -> b n d", b=B) 

        return x

    @property
    def proj_out_num(self):
        return self.pooling_out_num

class ablation_mlps_Projector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_num = 128
        self.pool_mpls = nn.Linear(2048*768, self.out_num*768)

        self.proj_mpls = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, x): 
        B = x.shape[0] # B*N*D

        x = rearrange(x, "b n d -> b (n d)") 
        x = self.pool_mpls(x)  
        x = rearrange(x, "b (n d) -> b n d", b=B) 

        x = rearrange(x, "b n d -> (b n) d") 
        x = self.projector(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B) 

        return x

    @property
    def proj_out_num(self):
        return self.out_num

import torch
import torch.nn as nn
from einops import rearrange
from transformers import BertTokenizer, BertModel
from torch.nn import MultiheadAttention

class ablation_qformerProjector(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_queries = 32
        self.num_heads = 8
        self.num_layers = 2
        self.query_embeds = nn.Parameter(torch.randn(self.num_queries, in_dim))
        
        self.self_attn = MultiheadAttention(embed_dim=in_dim, num_heads=self.num_heads)
        self.cross_attn = MultiheadAttention(embed_dim=in_dim, num_heads=self.num_heads)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_dim, nhead=self.num_heads)
            for _ in range(self.num_layers)
        ])
        
        self.proj_mpls = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        
        nn.init.xavier_uniform_(self.query_embeds)
        
    def forward(self, x):
        B, N, D = x.shape
        
        queries = self.query_embeds.unsqueeze(0).repeat(B, 1, 1) 
        
        queries = self.self_attn(
            query=queries,
            key=queries,
            value=queries
        )[0]
        
        x = x.permute(1, 0, 2).float() 
        queries = queries.permute(1, 0, 2).float()
        
        attended = self.cross_attn(
            query=queries, 
            key=x,
            value=x 
        )[0]
        
        for layer in self.layers:
            attended = layer(attended)
            
        output = self.proj_mpls(attended.permute(1, 0, 2)) 
        
        return output

    @property
    def proj_out_num(self):
        return self.num_queries


class SpatialPoolingProjector_med2e3(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.kernel_size = (1, 4, 4)
        self.slice_number = 32

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))] 

        self.in_dim = in_dim 
        self.out_dim = out_dim 
        
        self.projector_3d = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        self.projector_2d = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

        self.pooling_type = pooling_type


    def forward(self, x, slice_x, text_feature):
        B = x.shape[0] # B*N*D
        self.slice_number = slice_x.size(1)

        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x) 
            x = F.avg_pool3d(x, kernel_size=self.kernel_size) 
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x) 
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1) 
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1) 

        x = rearrange(x, "b n d -> (b n) d") 
        x = self.projector_3d(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B)  

        slice_x = self.projector_2d(slice_x)  # 
        slice_x = slice_x.view(B, self.slice_number, -1)

        feature_dim = x.size(-1) 

        feature_3d, feature_2d = x.clone(), slice_x.clone()
        feature_2d = feature_2d.unsqueeze(2) 
        feature_3d = feature_3d.view(B, self.num_patches_post[0], self.num_patches_post[1], self.num_patches_post[2], feature_dim) 
        feature_3d = feature_3d.unsqueeze(1).expand(B, self.slice_number//self.num_patches_post[0], self.num_patches_post[0], self.num_patches_post[1], self.num_patches_post[2], feature_dim).contiguous().view(B, self.slice_number, self.num_patches_post[1]*self.num_patches_post[2], feature_dim)
        
        feature_2d3d = torch.cat([feature_3d, feature_2d], dim=2)  
        feature_2d3d_pooling = feature_2d3d.mean(dim=2).squeeze(2)  
        valid_index_number = x.size(1) + slice_x.size(1) 

        text_feature_cleaned = text_feature[:, valid_index_number+1:, :].to(torch.bfloat16) 
        text_feature_cleaned = text_feature_cleaned.mean(dim=1)  
        slice_score = torch.matmul(feature_2d3d_pooling, text_feature_cleaned.unsqueeze(1).permute(0,2,1)).squeeze(2) 
        slice_score = F.softmax(slice_score, dim=1) 

        feature_2d_weighted = slice_x * slice_score.unsqueeze(2) 
        x = torch.cat([x, feature_2d_weighted], dim=1) 
        
        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num + self.slice_number


class SpatialPoolingProjector_m3d(nn.Module):
    def __init__(self, image_size, patch_size, in_dim, out_dim, layer_type, layer_num, pooling_type='spatial', pooling_size=2):
        super().__init__()
        self.in_dim = in_dim
        self.pooling_size = pooling_size
        self.kernel_size = (2, 2, 2)

        self.num_patches_pre = [img // pch for img, pch in zip(image_size, patch_size)]
        self.num_patches_post = [self.num_patches_pre[dim_index]//self.kernel_size[dim_index] for dim_index in range(len(self.num_patches_pre))]  

        if layer_type == 'linear':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        elif layer_type == 'mlp':
            depth = int(layer_num)
            modules = [nn.Linear(in_dim, out_dim)]
            for _ in range(1, depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(out_dim, out_dim))
            self.projector = nn.Sequential(*modules)
        else:
            print("Projector error!")

        self.pooling_type = pooling_type

    def forward(self, x):
        B = x.shape[0] # B*N*D

        if self.pooling_type == 'spatial':
            to_3d = Rearrange("b (p1 p2 p3) d -> b d p1 p2 p3", b=B, d=self.in_dim, p1=self.num_patches_pre[0], p2=self.num_patches_pre[1], p3=self.num_patches_pre[2])
            x = to_3d(x)
            x = F.avg_pool3d(x, kernel_size=self.kernel_size)
            to_seq = Rearrange("b d p1 p2 p3 -> b (p1 p2 p3) d", b=B, d=self.in_dim, p1=self.num_patches_post[0], p2=self.num_patches_post[1], p3=self.num_patches_post[2])
            x = to_seq(x) 
        elif self.pooling_type == 'sequence':
            x = x.permute(0, 2, 1) #b d n
            x = F.avg_pool1d(x, kernel_size=self.pooling_size**3, stride=self.pooling_size**3)
            x = x.permute(0, 2, 1) #b n d

        x = rearrange(x, "b n d -> (b n) d")
        x = self.projector(x) 
        x = rearrange(x, "(b n) d -> b n d", b=B)  

        return x

    @property
    def proj_out_num(self):
        num = 1
        for n in self.num_patches_post:
            num *= n
        return num