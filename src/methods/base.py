"""
Abstract base class for CoT Vector methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch

from ..models import CoTModelWrapper


class BaseCoTVectorMethod(ABC):
    """
    Abstract base class for CoT Vector methods.
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.dataset_type = dataset_type
        self.vector: Optional[torch.Tensor] = None
    
    @abstractmethod
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the CoT vector."""
        pass
