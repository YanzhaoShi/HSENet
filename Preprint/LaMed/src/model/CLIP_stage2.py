import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
from LaMed.src.utils.dist_utils import gather_features
import os
import json

loss_logs = []

class M3DCLIPConfig_stage2(PretrainedConfig):
    model_type = "m3d_clip_stage2"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        stage1_pretrained_CLIP_path = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        super().__init__(**kwargs)

import open_clip
current_step = 0

class M3DCLIP_stage2(PreTrainedModel):
    config_class = M3DCLIPConfig_stage2

    def __init__(self, config):
        super().__init__(config)

        # Stage2 2D-Enhanced 3D Vision Encoder (Trainable)
        self.vision_encoder = ViT_stage2(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            pos_embed=config.pos_embed,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
            classification=True,
        )

        # Stage2 Text Encoder (Trainable)
        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size) 
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)


        # Stage1 Pretrained CLIP (Frozen)
        self.stage1_pretrained_CLIP = AutoModel.from_pretrained(config.pretrained_model, trust_remote_code=True)
        self.stage1_pretrained_CLIP.requires_grad_(False)


        self.visual_encoder_2D = None
        if config.use_mask and config.use_2D_Encoder: 
            model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.visual_encoder_2D = model.visual.trunk
            self.visual_encoder_2D.requires_grad_(False)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

        self.mask_config = config

    def encode_image(self, image, images_2d, mask_ratio=None, text_features=None, image_path=None):
        image_feats, _ = self.vision_encoder(image, images_2d, mask_ratio, self.visual_encoder_2D, text_features, image_path) 

        image_feats = self.mm_vision_proj(image_feats) 
        image_feats = F.normalize(image_feats, dim=-1)  
        return image_feats

    def encode_text(self, input_id, attention_mask):
        text_feats_language = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"] 
        text_feats = self.mm_language_proj(text_feats_language) 
        text_feats = F.normalize(text_feats, dim=-1) 

        return text_feats, text_feats_language

    def forward(self, images, input_ids, attention_mask, labels, images_2d, global_step=None, epoch=None, **kwargs):
        mask_ratio = None
        image_path = None

        # Semantic Matching Loss
        with torch.inference_mode():
            text_features_stage1, text_features_language_stage1 = self.stage1_pretrained_CLIP.encode_text(input_ids, attention_mask)
            text_features_stage1 = text_features_stage1[:, 0]
            image_features_stage1 = self.stage1_pretrained_CLIP.encode_image(images, None, text_features_language_stage1)[:, 0]
        loss_CL_stage1, logits_per_image_stage1, logits_per_text_stage1 = self.image_text_contrastive_learning(image_features_stage1, text_features_stage1, labels)

        text_features, text_feats_language = self.encode_text(input_ids, attention_mask) 
        image_features = self.encode_image(images, images_2d, mask_ratio, text_features_language_stage1, image_path) 

        text_features = text_features[:, 0]
        image_features = image_features[:, 0]

        # CL Loss for image-text matching
        loss_CL_stage2, logits_per_image_stage2, logits_per_text_stage2 = self.image_text_contrastive_learning(image_features, text_features, labels)

        loss_relation = self.image_text_relation_regulation(logits_per_image_stage1.detach(), logits_per_text_stage1.detach(), logits_per_image_stage2, logits_per_text_stage2)

        relation_loss_weight = None
        max_weighted_step = 5000
        try:
            if global_step < max_weighted_step:
                relation_loss_weight = 0.1 * (1 - global_step / max_weighted_step)
            else:
                relation_loss_weight = 0.0
        except:
            relation_loss_weight = 0.0

        loss = loss_CL_stage2 + relation_loss_weight*loss_relation
        

        print_interval = 100  # 1
        try:
            if global_step % print_interval == 0 and global_step != 0:
                print()
                if global_step < max_weighted_step: 
                    print(f"Step {global_step}, loss: {loss}, loss_CL_stage1: {loss_CL_stage1}, loss_CL_stage2: {loss_CL_stage2}, loss_relation: {loss_relation}, relation_loss_weight: {relation_loss_weight}")
                else: 
                    print(f"Step {global_step}, loss: {loss}, loss_CL_stage1: {loss_CL_stage1}, loss_CL_stage2: {loss_CL_stage2}, loss_relation: {loss_relation}")
        except:
            pass
        ret = {
            "loss": loss,
            "logits": (logits_per_image_stage2 + logits_per_text_stage2) / 2.0,
        }

        return ret

    def image_text_contrastive_learning(self, image_features, text_features, labels):
        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T  
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        return loss, logits_per_image, logits_per_text

    def image_text_relation_regulation(self, logits_per_image, logits_per_text, logits_per_image_masked, logits_per_text_masked):
        loss_per_image = F.mse_loss(logits_per_image, logits_per_image_masked)
        loss_per_text = F.mse_loss(logits_per_text, logits_per_text_masked)
        return (loss_per_image + loss_per_text) / 2




AutoConfig.register("m3d_clip_stage2", M3DCLIPConfig_stage2)
AutoModel.register(M3DCLIPConfig_stage2, M3DCLIP_stage2)