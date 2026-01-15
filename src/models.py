"""
Model wrapper with hook mechanisms for CoT Vector injection and extraction.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Callable


class CoTModelWrapper(nn.Module):
    """
    Wrapper around HuggingFace models that provides:
    1. Forward hooks for extracting activations
    2. Injection hooks for adding CoT vectors
    """
    
    def __init__(self, model_path: str, model_name: str = "qwen"):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        
        # Load model with multi-GPU support
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = self._get_num_layers()
        self.hidden_size = self._get_hidden_size()
        
        # Hook management
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, torch.Tensor] = {}
        self._injection_vector_cached: Optional[torch.Tensor] = None
        
    def _get_num_layers(self) -> int:
        if self.model_name == "qwen":
            return len(self.model.model.layers)
        elif self.model_name == "llama":
            return len(self.model.model.layers)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _get_hidden_size(self) -> int:
        return self.model.config.hidden_size
    
    def _get_layer(self, layer_idx: int) -> nn.Module:
        if self.model_name in ["qwen", "llama"]:
            return self.model.model.layers[layer_idx]
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def register_extraction_hook(
        self, 
        layer_idx: int, 
        position_ids: Optional[torch.Tensor] = None,
        requires_grad: bool = False
    ):
        """
        Register hook to extract activations at specified layer.
        
        Args:
            layer_idx: Layer index to extract from
            position_ids: Optional position indices to extract
            requires_grad: If True, keep gradients for training. If False, detach.
        """
        layer = self._get_layer(layer_idx)
        
        def hook_fn(module, input, output):
            # output is tuple (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Extract at specified positions or all
            if position_ids is not None:
                extracted = hidden_states[:, position_ids, :]
            else:
                extracted = hidden_states
            
            # Convert to float32 for numerical stability
            if requires_grad:
                # Keep gradients for training - don't detach
                self._activations[layer_idx] = extracted.float()
            else:
                # Detach for inference to save memory
                self._activations[layer_idx] = extracted.detach().float()
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def register_injection_hook(
        self, 
        layer_idx: int, 
        vector: torch.Tensor, 
        scaling_factor: float = 1.0,
        requires_grad: bool = False
    ):
        """
        Register hook to inject CoT vector at specified layer.
        Pre-caches the vector in correct dtype/device for efficiency.
        """
        layer = self._get_layer(layer_idx)
        
        # Get target device and dtype
        target_device = next(layer.parameters()).device
        target_dtype = next(layer.parameters()).dtype
        
        if requires_grad:
            # For training: store the original vector reference to maintain gradient connection
            # We'll do dtype conversion in the hook to maintain the computation graph
            self._injection_vector_raw = vector
            self._injection_scaling_factor = scaling_factor
        else:
            # For inference: pre-convert and cache for efficiency
            vector_scaled = scaling_factor * vector.to(device=target_device, dtype=target_dtype)
            if vector_scaled.dim() == 1:
                vector_scaled = vector_scaled.unsqueeze(0).unsqueeze(0)
            elif vector_scaled.dim() == 2:
                vector_scaled = vector_scaled.unsqueeze(0)
            self._injection_vector_cached = vector_scaled
        
        # Store flags
        self._injection_requires_grad = requires_grad
        self._injection_target_device = target_device
        self._injection_target_dtype = target_dtype
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None
            
            if self._injection_requires_grad:
                # Training: compute on-the-fly to maintain gradient flow
                vec = self._injection_scaling_factor * self._injection_vector_raw
                vec = vec.to(device=hidden_states.device, dtype=hidden_states.dtype)
                if vec.dim() == 1:
                    vec = vec.unsqueeze(0).unsqueeze(0)
                elif vec.dim() == 2:
                    vec = vec.unsqueeze(0)
                modified = hidden_states + vec.expand_as(hidden_states)
            else:
                # Inference: use pre-cached vector
                modified = hidden_states + self._injection_vector_cached.expand_as(hidden_states)
            
            if rest is not None:
                return (modified,) + rest
            return modified
        
        handle = layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)
        return handle
    
    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get extracted activations for a layer."""
        return self._activations.get(layer_idx)
    
    def clear_hooks(self):
        """Remove all registered hooks and clear cached data."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._activations.clear()
        
        # Clear injection-related attributes
        self._injection_vector_cached = None
        if hasattr(self, '_injection_vector_raw'):
            self._injection_vector_raw = None
        if hasattr(self, '_injection_scaling_factor'):
            self._injection_scaling_factor = None
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, input_ids: torch.Tensor, **kwargs):
        """Generate text."""
        return self.model.generate(input_ids=input_ids, **kwargs)
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property 
    def dtype(self):
        return next(self.model.parameters()).dtype


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """Load tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
