from models.projectors import ProjectorConfig, VisionProjector
from models.qwen import Qwen2Config, Qwen2ForCausalLM
from models.vision_encoder import VisionConfig, VisionEncoder

import torch
from safetensors.torch import load_file

def load_models(paths: list[str], device: str = "cuda"):
    """
    Load vision encoder, projector, and language model from safetensors shards
    """
    # 1. Initialize configs
    vision_config = VisionConfig()
    llm_config = Qwen2Config(
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=2,
    )
    projector_config = ProjectorConfig(
        vision_size=1024,
        language_size=2048,
        downsample_factor=0.5,
    )
    
    # 2. Initialize models
    vision_encoder = VisionEncoder(vision_config)
    llm = Qwen2ForCausalLM(llm_config)
    projector = VisionProjector(projector_config)
    
    # 3. Load weights once from all shards
    state_dict = {}
    for path in paths:
        shard = load_file(path)
        state_dict.update(shard)
    
    # 4. Split and remap for each component
    vision_weights = {}
    llm_weights = {}
    projector_weights = {}
    
    for k, v in state_dict.items():
        if k.startswith('vision_model.'):
            k = k[len('vision_model.'):]
            if 'class_embedding' in k:
                k = k.replace('class_embedding', 'cls_token')
            elif 'patch_embedding' in k:
                k = k.replace('patch_embedding', 'patch_embed')
            elif 'position_embedding' in k:
                k = k.replace('position_embedding', 'pos_embed')
            if k.startswith('encoder.'):
                k = k[len('encoder.'):]
            vision_weights[k] = v
            
        elif k.startswith('language_model.'):
            if k.startswith('language_model.model.'):
                k = k[len('language_model.model.'):]
            else:
                k = k[len('language_model.'):]
            
            if not k.startswith('lm_head'):
                k = f'model.{k}'
                k = k.replace('self_attn', 'attention')
            llm_weights[k] = v
            
        elif k.startswith('mlp1.'):
            k = k.replace('mlp1.', 'layers.')
            projector_weights[k] = v
    
    # 5. Load weights with verification
    vision_encoder.load_state_dict(vision_weights, strict=True)
    llm.load_state_dict(llm_weights, strict=True)
    projector.load_state_dict(projector_weights, strict=True)
    
    # 6. Move to device and set eval
    vision_encoder = vision_encoder.to(device).eval()
    projector = projector.to(device).eval()
    llm = llm.to(device).eval()
    
    return vision_encoder, projector, llm