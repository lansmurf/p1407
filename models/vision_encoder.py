import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
from typing import Optional

@dataclass
class VisionConfig:
    hidden_size: int = 1024
    image_size: int = 448  
    patch_size: int = 14
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    layer_scale_init_value: float = 0.1  # MetaFormer-style init
    norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    proj_dropout: float = 0.0
    path_dropout: float = 0.1  # Stochastic depth

class VisionEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding with proper initialization
        self.patch_embed = nn.Conv2d(
            3, config.hidden_size, 
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True  # InternVL uses bias here
        )
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        num_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]
        
        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        return x

class Attention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5
        
        # QKV projection with bias
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use scaled_dot_product_attention for efficiency
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.proj_dropout)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig, drop_path: float = 0.):
        super().__init__()
        # Normalization layers with bias
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=True)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=True)
        
        # Attention and MLP
        self.attn = Attention(config)
        self.mlp = MLP(config)
        
        # Layer scale parameters (from MetaFormer)
        scale_init = config.layer_scale_init_value
        self.ls1 = nn.Parameter(torch.ones(config.hidden_size) * scale_init)
        self.ls2 = nn.Parameter(torch.ones(config.hidden_size) * scale_init)
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm design with layer scaling
        x = x + self.drop_path(self.ls1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = VisionEmbedding(config)
        
        # Progressive drop path rates
        dpr = [x.item() for x in torch.linspace(0, config.path_dropout, config.num_hidden_layers)]
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(config, drop_path=dpr[i])
            for i in range(config.num_hidden_layers)
        ])
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        x = self.embeddings(pixel_values)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return x