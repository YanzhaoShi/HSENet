import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
from LaMed.src.model.multimodal_encoder.vit import ViT
from LaMed.src.utils.dist_utils import gather_features
import os
import json

loss_logs = []

class M3DCLIPConfig(PretrainedConfig):
    model_type = "m3d_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "",
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


def update_mask_ratio(training_step, initial_mask_ratio=0.05, max_mask_ratio=40.0, temperature_factor=0.01):
    """
    Update the mask ratio using a Gaussian temperature function based on training step.
    
    Args:
        training_step (int): The current training step.
        initial_mask_ratio (float): The initial mask ratio, default is 0.05.
        max_mask_ratio (float): The maximum value the mask ratio can reach, default is 40.0.
        temperature_factor (float): The factor controlling the growth rate of the mask ratio.
        
    Returns:
        float: The updated mask ratio.
    """
    growth = math.exp(- (training_step * temperature_factor)**2) 
    new_mask_ratio = initial_mask_ratio + (max_mask_ratio - initial_mask_ratio) * (1 - growth)
    
    if new_mask_ratio > max_mask_ratio:
        return max_mask_ratio
    return new_mask_ratio

import open_clip
current_step = 0

class M3DCLIP(PreTrainedModel):
    config_class = M3DCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = ViT(
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
        self.visual_encoder_2D = None
        if config.use_mask and config.use_2D_Encoder: 
            model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.visual_encoder_2D = model.visual.trunk
            self.visual_encoder_2D.requires_grad_(False)

        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size) 
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

        self.mask_config = config

    def encode_image(self, image, mask_ratio=None, text_features=None, image_path=None):
        image_feats, _, image_feats_masked, _ = self.vision_encoder(image, mask_ratio, self.visual_encoder_2D, text_features, image_path) 

        image_feats = self.mm_vision_proj(image_feats) 
        if mask_ratio != None: 
            image_feats_masked = self.mm_vision_proj(image_feats_masked)

        image_feats = F.normalize(image_feats, dim=-1)
        if mask_ratio != None: 
            image_feats_masked = F.normalize(image_feats_masked, dim=-1)

        if mask_ratio == None: 
            return image_feats
        else:
            return image_feats, image_feats_masked

    def encode_text(self, input_id, attention_mask):
        text_feats_language = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"] 
        text_feats = self.mm_language_proj(text_feats_language) 
        text_feats = F.normalize(text_feats, dim=-1) 

        return text_feats, text_feats_language

    def forward(self, images, input_ids, attention_mask, labels, global_step=None, epoch=None, **kwargs):
        if global_step is None: 
            print("global_step is None in forward")
            global_step = 0
        if self.mask_config.use_mask:
            max_mask_ratio = 0.400
            temperature_factor = 1e-4
            mask_ratio = update_mask_ratio(global_step, initial_mask_ratio=self.mask_config.mask_rate, max_mask_ratio=max_mask_ratio, temperature_factor=temperature_factor) 
        else:
            mask_ratio = None
        text_features, text_feats_language = self.encode_text(input_ids, attention_mask) 
        image_features, image_features_masked = self.encode_image(images, mask_ratio, text_feats_language) 

        text_features = text_features[:, 0]
        image_features = image_features[:, 0]
        image_features_masked = image_features_masked[:, 0]
        
        # CL Loss for image-text matching
        loss_unmasked, logits_per_image, logits_per_text = self.image_text_contrastive_learning(image_features, text_features, labels)

        # CL Loss for image-text matching with masked image features
        loss_masked, logits_per_image_masked, logits_per_text_masked = self.image_text_contrastive_learning(image_features_masked, text_features, labels)

        loss_relation = self.image_text_relation_regulation(logits_per_image.detach(), logits_per_text.detach(), logits_per_image_masked, logits_per_text_masked)
        loss = loss_unmasked + 0.1*loss_masked
        print_interval = 100 

        if global_step % print_interval == 0 and global_step != 0:
            print()
            print(f"Step {global_step}, mask_ratio: {mask_ratio} - loss: {loss}, loss_unmasked: {loss_unmasked}, loss_masked: {loss_masked}, loss_relation: {loss_relation}")

        ret = {
            "loss": loss,
            "logits": (logits_per_image + logits_per_text) / 2.0,
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




AutoConfig.register("m3d_clip", M3DCLIPConfig)
AutoModel.register(M3DCLIPConfig, M3DCLIP)