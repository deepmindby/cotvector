"""
Extracted CoT Vector implementation.
Implements Eq. 4 and 5 from the paper.
"""

import torch
from typing import List, Optional
from tqdm import tqdm

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class ExtractedCoTVector(BaseCoTVectorMethod):
    """
    Extract CoT Vector by computing activation differences.
    
    v_CoT = (1/N) * Σ (α_CoT(a) - α_NonCoT(a))
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def _get_answer_positions(self, full_ids: torch.Tensor, qa_ids: torch.Tensor) -> torch.Tensor:
        """Get position indices of answer tokens."""
        full_len = full_ids.shape[1]
        qa_len = qa_ids.shape[1]
        
        # Answer tokens are at the end (after Q+CoT vs after Q)
        # For CoT input: positions from qa_len to full_len are answer positions relative to the beginning
        # But we want positions in the full sequence
        answer_start = full_len - (qa_len - self._get_question_length(qa_ids))
        positions = torch.arange(answer_start, full_len, device=full_ids.device)
        
        return positions
    
    def _get_question_length(self, qa_ids: torch.Tensor) -> int:
        """Estimate question length (used for position calculation)."""
        # This is approximate - we just need the answer token positions
        return qa_ids.shape[1] // 2
    
    def extract_single(self, sample) -> torch.Tensor:
        """Extract CoT vector from a single sample."""
        device = self.model_wrapper.device
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            cot_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            non_cot_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize
        cot_encoding = self.tokenizer(cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        non_cot_encoding = self.tokenizer(non_cot_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        cot_ids = cot_encoding["input_ids"].to(device)
        non_cot_ids = non_cot_encoding["input_ids"].to(device)
        cot_mask = cot_encoding["attention_mask"].to(device)
        non_cot_mask = non_cot_encoding["attention_mask"].to(device)
        
        # Find answer token positions (last N tokens where N = answer length)
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        answer_len = answer_ids.shape[1]
        
        # Get positions for answer tokens
        cot_answer_pos = list(range(cot_ids.shape[1] - answer_len, cot_ids.shape[1]))
        non_cot_answer_pos = list(range(non_cot_ids.shape[1] - answer_len, non_cot_ids.shape[1]))
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        # Extract CoT activations
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(cot_ids, attention_mask=cot_mask)
        cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        cot_answer_activation = cot_activation[:, cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Clear and extract non-CoT activations
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_extraction_hook(self.layer_idx)
        with torch.no_grad():
            self.model_wrapper(non_cot_ids, attention_mask=non_cot_mask)
        non_cot_activation = self.model_wrapper.get_activations(self.layer_idx)
        non_cot_answer_activation = non_cot_activation[:, non_cot_answer_pos, :].mean(dim=1)  # [1, hidden]
        
        # Compute difference
        diff = cot_answer_activation - non_cot_answer_activation  # [1, hidden]
        
        self.model_wrapper.clear_hooks()
        
        return diff.squeeze(0)  # [hidden]
    
    def extract(self, support_samples: List) -> torch.Tensor:
        """
        Extract task-general CoT vector from support set.
        Implements Eq. 5: v_E = (1/N) * Σ v_CoT,i
        """
        print(f"Extracting CoT vectors from {len(support_samples)} samples at layer {self.layer_idx}...")
        
        vectors = []
        for sample in tqdm(support_samples, desc="Extracting", ncols=100):
            try:
                vec = self.extract_single(sample)
                vectors.append(vec)
            except Exception as e:
                # Skip problematic samples
                continue
        
        if not vectors:
            raise ValueError("No vectors extracted!")
        
        # Average all vectors (Eq. 5)
        stacked = torch.stack(vectors, dim=0)  # [N, hidden]
        task_vector = stacked.mean(dim=0)  # [hidden]
        
        self.vector = task_vector
        
        print(f"Extracted vector: shape={task_vector.shape}, norm={task_vector.norm().item():.4f}")
        
        return task_vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        return self.vector
