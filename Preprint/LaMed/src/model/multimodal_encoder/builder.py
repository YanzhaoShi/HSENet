from .vit import ViT3DTower, ViT3DTower_dual_encoders


def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    if 'vit3d' in vision_tower.lower():
        return ViT3DTower(config, **kwargs)
    elif 'vit_stage2_dual_encoders' in vision_tower.lower():
        return ViT3DTower_dual_encoders(config, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')