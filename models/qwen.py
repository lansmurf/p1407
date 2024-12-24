import torch
from torch import nn
from typing import Optional, Tuple, Dict, Any
import math

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class Qwen2Config:
    def __init__(
        self,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,
        num_hidden_layers: int = 36,
        intermediate_size: int = 11008,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        vocab_size: int = 151674,
        sliding_window: int = 32768,
        attention_dropout: float = 0.0,
        rope_theta: float = 1000000.0,
        use_sliding_window: bool = False,
        _attn_implementation: str = "flash_attention_2",
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self._attn_implementation = _attn_implementation

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb_dim = head_dim
        
        # Create inverse frequency buffer
        theta = config.rope_theta
        pos_embd_idx = torch.arange(head_dim)
        freqs = 1.0 / (theta ** (pos_embd_idx.float() / head_dim))
        self.register_buffer("freqs", freqs, persistent=False)
        
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = self.freqs[None, None, :]
        t = position_ids[:, :, None].float() * freqs
        cos = t.cos()
        sin = t.sin()
        return cos, sin

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_length, self.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Apply rotary embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat KV heads if needed
        if self.num_key_value_groups > 1:
            key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=1)
            value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=1)

        # Flash Attention path
        if self.config._attn_implementation == "flash_attention_2":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=attention_mask is None,
            )
        else:
            # Classical attention path
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
                
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = torch.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Block(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.attention = Qwen2Attention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            
        # Get rotary embeddings
        position_embeddings = self.rotary_emb(position_ids)
        
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            
        return self.norm(hidden_states)

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        return {
            "loss": loss,
            "logits": logits,
        }