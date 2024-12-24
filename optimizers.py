import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import functools

def create_optimizer(model, lr=1e-4, weight_decay=0.1, beta1=0.9, beta2=0.95):
    """Creates AdamW optimizer with standard hyperparameters"""
    return AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay
    )

def warmup_stable_decay(current_step: int, *, total_steps: int, warmup_frac: float, stable_frac: float) -> float:
    """Learning rate schedule with linear warmup, stable period, and linear decay
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        warmup_frac: Fraction of total steps for warmup
        stable_frac: Fraction of total steps for stable LR
    """
    warmup_steps = int(total_steps * warmup_frac)
    stable_steps = int(total_steps * stable_frac)
    
    if warmup_frac + stable_frac >= 1.0:
        raise ValueError("warmup_frac + stable_frac must be < 1.0 to allow decay period")
        
    if current_step < warmup_steps:
        # Linear warmup (add 1 to avoid division by zero)
        return float(current_step + 1) / (warmup_steps + 1)
    
    elif current_step < (warmup_steps + stable_steps):
        return 1.0
        
    else:
        # Linear decay
        decay_steps = total_steps - warmup_steps - stable_steps
        decay_step = current_step - warmup_steps - stable_steps
        return max(0.0, 1 - (decay_step / decay_steps))

def create_scheduler(optimizer, total_steps: int, warmup_frac=0.01, stable_frac=0.6):
    """Creates a learning rate scheduler with warmup, stable period and decay"""
    lr_lambda = functools.partial(
        warmup_stable_decay,
        total_steps=total_steps,
        warmup_frac=warmup_frac,
        stable_frac=stable_frac
    )
    return LambdaLR(optimizer, lr_lambda=lr_lambda)