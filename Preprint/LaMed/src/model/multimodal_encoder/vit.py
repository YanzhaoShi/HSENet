
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
import open_clip

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class regular_attention(nn.Module):

    def __init__(self, in_channels=16, out_channels=8, emb_dim=768, output_dim=768, dropout=0.1, aropout=0.0):
        super(regular_attention, self).__init__()
        self.emb_dim = emb_dim
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
        self.attn = None
        self.output_linear = nn.Linear(emb_dim,emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(emb_dim)
    
    def forward(self, Query, Key, Value, context=None, mask=None):
        '''
        :param x: [1-4, 32, 768]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        ''' 
        query_list=self.Wq(Query)
        key_list=self.Wk(Key)
        value_list=self.Wv(Value)
        x, self.attn = attention(query_list, key_list, value_list, mask=mask, dropout=self.dropout)
        x = self.output_linear(x)
        x = self.norm(query_list + self.dropout_2(x)) 

        return x, self.attn


class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )


        self.patch_score_proj = nn.Linear(hidden_size, 1)
        self.patch_score_norm = nn.Sigmoid()
        self.slice_guided_attention = regular_attention()
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_masked = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, k=None, visual_encoder_2D=None, text_features=None, image_path=None): 
        batch, slice_num = x.size(0), x.size(2)
        hidden_states_out_masked = None
        x_masked = None

        abaltion_score_method = "CrossAttention" 
        abaltion_score_feature = "2DSlice"

        if visual_encoder_2D is not None and not abaltion_score_method == "Random": 
            x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False) 
            x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
            x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
            slice_features = visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())
        if k != None:
            batch_size, num_patches, slice_num, dim = x.size(0), x.size(1), 32, x.size(2)
            unmasked_number = int(num_patches * (1-k))
            if abaltion_score_method == "CrossAttention":
                if visual_encoder_2D is not None and abaltion_score_feature == "2DSlice":
                    semantic_features = slice_features.view(batch, slice_num, -1)

            if abaltion_score_method == "CrossAttention":
                patch_score = self.slice_guided_attention(x, semantic_features, semantic_features)
                patch_score = self.patch_score_proj(patch_score)
                scores_flat = self.patch_score_norm(patch_score.view(batch_size, num_patches))

            image_feats_spacial = x.clone()
            image_feats_spacial = image_feats_spacial * scores_flat.unsqueeze(-1)
            topk_scores, topk_indices = torch.topk(scores_flat, k=unmasked_number, dim=1, largest=True, sorted=False)
            sorted_topk_indices, _ = torch.sort(topk_indices, dim=1)
            topk_features = torch.gather(image_feats_spacial.clone(), dim=1, index=sorted_topk_indices.unsqueeze(-1).expand(-1, -1, 768)) 
            x_masked = topk_features

            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
                x_masked = torch.cat((cls_token, x_masked), dim=1)

            hidden_states_out_masked = []
            for blk in self.blocks:  # 12 x TransformerBlock
                x_masked = blk(x_masked)
                hidden_states_out_masked.append(x_masked)
            x_masked = self.norm_masked(x_masked)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if k != None:
            return x, hidden_states_out, x_masked, hidden_states_out_masked
        else:
            return x, hidden_states_out


class ViT_stage2(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )


        self.patch_score_proj = nn.Linear(hidden_size, 1)
        self.patch_score_norm = nn.Sigmoid()
        self.slice_guided_attention = regular_attention()
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, image_2d, k=None, visual_encoder_2D=None, text_features=None, image_path=None):
        batch, slice_num = x.size(0), x.size(2)
        hidden_states_out_masked = None
        x_masked = None

        abaltion_score_method = "CrossAttention"
        abaltion_score_feature = "2DSlice"

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())

        if True:
            batch_size, num_patches, slice_num, dim = x.size(0), x.size(1), 32, x.size(2)

            if abaltion_score_method == "CrossAttention":
                if abaltion_score_feature == "2DSlice":
                    semantic_features = image_2d.to(x.device).view(batch, slice_num, -1)

            if abaltion_score_method == "CrossAttention":
                patch_score, att_map = self.slice_guided_attention(x, semantic_features, semantic_features)
                score_strategy =  "linear_sigmoid"  # "linear_sigmoid" "linear" "att_map"
                if score_strategy == "linear_sigmoid":
                    patch_score = self.patch_score_proj(patch_score) 
                    scores_flat = self.patch_score_norm(patch_score.view(batch_size, num_patches))
                elif score_strategy == "linear":
                    scores_flat = patch_score.view(batch_size, num_patches)
                elif score_strategy == "att_map": 
                    scores_flat = att_map.sum(-1).view(batch_size, num_patches) 

            x_weighted = x * scores_flat.unsqueeze(-1)

            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x_weighted.shape[0], -1, -1)
                x_weighted = torch.cat((cls_token, x_weighted), dim=1)

            hidden_states_out_weighted = []
            for blk in self.blocks:  # 12 x TransformerBlock
                x_weighted = blk(x_weighted)
                hidden_states_out_weighted.append(x_weighted)
            x_weighted = self.norm(x_weighted)

        return x_weighted, hidden_states_out_weighted


class ViT_stage1(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, k=None, visual_encoder_2D=None, text_features=None, image_path=None):
        batch, slice_num = x.size(0), x.size(2)
        hidden_states_out_masked = None

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())

        batch_size, num_patches, dim = x.size(0), x.size(1), x.size(2)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out_masked = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out_masked.append(x)
        x = self.norm(x)

        return x, hidden_states_out_masked

class ViT4LLM_v3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.patch_score_proj = nn.Linear(hidden_size, 1)
        self.patch_score_norm = nn.Sigmoid()
        self.slice_guided_attention = regular_attention()
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_masked = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.visual_encoder_2D = model.visual.trunk
        self.visual_encoder_2D.requires_grad_(False)

    def forward(self, x):
        batch, slice_num = x.size(0), x.size(2)

        x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False)
        x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
        x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
        slice_features = self.visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone()) 

        batch_size, num_patches, slice_num, dim = x.size(0), x.size(1), 32, x.size(2)

        semantic_features = slice_features.view(batch, slice_num, -1)

        patch_score = self.slice_guided_attention(x, semantic_features, semantic_features) 

        patch_score = self.patch_score_proj(patch_score) 
        scores_flat = self.patch_score_norm(patch_score.view(batch_size, num_patches))

        image_feats_spacial = x.clone()
        image_feats_spacial = image_feats_spacial * scores_flat.unsqueeze(-1)
        x_masked = image_feats_spacial

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
            x_masked = torch.cat((cls_token, x_masked), dim=1)

        hidden_states_out_masked = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x_masked = blk(x_masked)
            hidden_states_out_masked.append(x_masked)
        x_masked = self.norm_masked(x_masked)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        return x, hidden_states_out, x_masked, hidden_states_out_masked 


class ViT4LLM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out


class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT4LLM(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )


    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

class ViT3DTower_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'both'
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower = ViT4LLM_v3(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states, weighted_last_feature, weighted_hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature 
            weighted_image_features = weighted_last_feature 
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
            weighted_image_features = weighted_hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch': 
            image_features = image_features[:, 1:] 
            weighted_image_features = weighted_image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            weighted_image_features = weighted_image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if self.remain_2d3d_ViT_type == "both":
            return image_features, weighted_image_features
        elif self.remain_2d3d_ViT_type == "3dvit":
            return image_features
        elif self.remain_2d3d_ViT_type == "2d3dvit":
            return weighted_image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size



class ViT4LLM_v3_med2e3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.visual_encoder_2D = model.visual.trunk
        self.visual_encoder_2D.requires_grad_(False)

    def forward(self, x):
        batch, slice_num = x.size(0), x.size(2)

        x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False)
        x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
        x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
        slice_features = self.visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())

        batch_size, num_patches, dim = x.size(0), x.size(1), x.size(2)
        slice_features = slice_features.view(batch, slice_num, -1)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        return x, hidden_states_out, slice_features


class ViT3DTower_med2e3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'both'
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower = ViT4LLM_v3_med2e3(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states, slice_last_feature = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature 
            slice_last_feature = slice_last_feature 
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
            slice_last_feature = slice_last_feature[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:] 
            slice_last_feature = slice_last_feature
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            slice_last_feature = slice_last_feature
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features, slice_last_feature

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size


class ViT3DTower_dual_encoders(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load dual_encoder seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'dual_vits'
            print("Load dual_encoder seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower_stage1 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

        self.vision_tower_stage2 = ViT_stage2(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):

        last_feature_stage1, hidden_states_stage1 = self.vision_tower_stage1(images)
        last_feature_stage2, hidden_states_stage2 = self.vision_tower_stage2(images, images_2d)

        image_features_stage1 = last_feature_stage1
        image_features_stage2 = last_feature_stage2

        if self.select_feature == 'patch':
            image_features_stage1 = image_features_stage1[:, 1:]
            image_features_stage2 = image_features_stage2[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features_stage1 = image_features_stage1
            image_features_stage2 = image_features_stage2
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if self.remain_2d3d_ViT_type == "dual_vits":
            return image_features_stage1, image_features_stage2
        elif self.remain_2d3d_ViT_type == "3d_vit":
            return image_features_stage1
        elif self.remain_2d3d_ViT_type == "2e3_vit":
            return image_features_stage2

    @property
    def dtype(self):
        return self.vision_tower_stage1.dtype

    @property
    def device(self):
        return self.vision_tower_stage1.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage1.hidden_size

class ViT3DTower_3dvit_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_stage1 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):
        
        last_feature_stage1, hidden_states_stage1 = self.vision_tower_stage1(images)

        image_features_stage1 = last_feature_stage1

        if self.select_feature == 'patch':
            image_features_stage1 = image_features_stage1[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features_stage1 = image_features_stage1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features_stage1

    @property
    def dtype(self):
        return self.vision_tower_stage1.dtype

    @property
    def device(self):
        return self.vision_tower_stage1.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage1.hidden_size


class ViT3DTower_2e3vit_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_stage2 = ViT_stage2(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):

        last_feature_stage2, hidden_states_stage2 = self.vision_tower_stage2(images, images_2d)

        image_features_stage2 = last_feature_stage2

        if self.select_feature == 'patch': 
            image_features_stage2 = image_features_stage2[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features_stage2 = image_features_stage2
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features_stage2

    @property
    def dtype(self):
        return self.vision_tower_stage2.dtype

    @property
    def device(self):
        return self.vision_tower_stage2.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage2.hidden_size


class ViT3DTower_reproduce_med2e3_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_reproduce_med2e3 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):
        
        last_feature, hidden_states = self.vision_tower_reproduce_med2e3(images)

        image_features = last_feature

        if self.select_feature == 'patch': 
            image_features = image_features[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features, images_2d 

    @property
    def dtype(self):
        return self.vision_tower_reproduce_med2e3.dtype

    @property
    def device(self):
        return self.vision_tower_reproduce_med2e3.device

    @property
    def hidden_size(self):
        return self.vision_tower_reproduce_med2e3.hidden_size