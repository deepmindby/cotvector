"""
Evaluation logic for CoT Vectors.
"""

import torch
import re
from typing import Optional, List, Dict, Any
from tqdm import tqdm
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .models import CoTModelWrapper, load_tokenizer
from .data_utils import PROMPT_TEMPLATES
from .utils import extract_answer_from_text, compare_answers


class AnswerStoppingCriteria(StoppingCriteria):
    """Stop generation when answer pattern is detected."""
    
    def __init__(self, tokenizer, dataset_type: str = "gsm8k", min_tokens: int = 30):
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.min_tokens = min_tokens
        self.generated_tokens = 0
        
        # Patterns for different datasets
        if dataset_type == "mmlu_pro":
            self.patterns = [
                re.compile(r'\\boxed\s*\{[A-J]\}'),
                re.compile(r'[Tt]he\s+answer\s+is\s*:?\s*\(?([A-J])\)?'),
            ]
        else:
            self.patterns = [
                re.compile(r'\\boxed\s*\{[^{}]+\}'),
                re.compile(r'####\s*[\d,]+'),
            ]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.generated_tokens += 1
        if self.generated_tokens < self.min_tokens:
            return False
        
        # Check last 150 tokens for answer pattern
        text = self.tokenizer.decode(input_ids[0, -150:], skip_special_tokens=True)
        return any(p.search(text) for p in self.patterns)
    
    def reset(self):
        self.generated_tokens = 0


class CoTEvaluator:
    """Evaluator for CoT Vector experiments."""
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        dataset_type: str = "gsm8k",
        max_new_tokens: int = 512,
        num_beams: int = 3,
        temperature: float = 1.0,
        do_sample: bool = False,
        use_early_stopping: bool = False,
    ):
        self.model_wrapper = model_wrapper
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        self.do_sample = do_sample
        
        # Early stopping
        self.stopping_criteria = None
        if use_early_stopping:
            self.stopping_criteria = StoppingCriteriaList([
                AnswerStoppingCriteria(tokenizer, dataset_type)
            ])
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": 1.0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if num_beams > 1:
            gen_kwargs["length_penalty"] = 0.0
        
        self.generation_config = GenerationConfig(**gen_kwargs)
        
        # Get prompt template
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def evaluate_sample(
        self,
        sample,
        vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        scaling_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluate a single sample."""
        # Build prompt
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        
        # Tokenize
        device = self.model_wrapper.device
        encoding = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        input_len = input_ids.shape[1]
        
        # Clear previous hooks
        self.model_wrapper.clear_hooks()
        
        # Register injection hook if vector provided
        if vector is not None and layer_idx is not None:
            self.model_wrapper.register_injection_hook(layer_idx, vector, scaling_factor)
        
        # Reset stopping criteria
        if self.stopping_criteria:
            for sc in self.stopping_criteria:
                if hasattr(sc, 'reset'):
                    sc.reset()
        
        # Generate
        with torch.no_grad():
            outputs = self.model_wrapper.model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=self.generation_config,
                stopping_criteria=self.stopping_criteria,
            )
        
        # Decode output
        generated_ids = outputs[0, input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract answer
        predicted = extract_answer_from_text(generated_text, self.dataset_type)
        is_correct = compare_answers(predicted, sample.answer, self.dataset_type)
        
        # Clear hooks
        self.model_wrapper.clear_hooks()
        
        return {
            "predicted": predicted,
            "ground_truth": sample.answer,
            "correct": is_correct,
            "generated_text": generated_text,
            "num_tokens": len(generated_ids),
        }
    
    def evaluate_dataset(
        self,
        samples: List,
        vector: Optional[torch.Tensor] = None,
        layer_idx: Optional[int] = None,
        scaling_factor: float = 1.0,
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Evaluate on a dataset."""
        correct = 0
        total = len(samples)
        results = []
        
        pbar = tqdm(samples, desc=desc, ncols=100)
        for sample in pbar:
            result = self.evaluate_sample(sample, vector, layer_idx, scaling_factor)
            results.append(result)
            
            if result["correct"]:
                correct += 1
            
            # Update progress bar
            acc = correct / len(results) * 100
            pbar.set_postfix({"acc": f"{acc:.1f}%"})
        
        accuracy = correct / total * 100
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }


def run_baseline_evaluation(
    model_wrapper: CoTModelWrapper,
    tokenizer,
    test_samples: List,
    dataset_type: str,
    max_new_tokens: int = 512,
    num_beams: int = 3,
    use_early_stopping: bool = False,
) -> Dict[str, Any]:
    """Run baseline evaluation without CoT vector."""
    evaluator = CoTEvaluator(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        use_early_stopping=use_early_stopping,
    )
    
    return evaluator.evaluate_dataset(test_samples, desc="Baseline")


def run_injection_evaluation(
    model_wrapper: CoTModelWrapper,
    tokenizer,
    test_samples: List,
    vector: torch.Tensor,
    layer_idx: int,
    dataset_type: str,
    scaling_factor: float = 1.0,
    max_new_tokens: int = 512,
    num_beams: int = 3,
    use_early_stopping: bool = False,
) -> Dict[str, Any]:
    """Run evaluation with CoT vector injection."""
    evaluator = CoTEvaluator(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        dataset_type=dataset_type,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        use_early_stopping=use_early_stopping,
    )
    
    return evaluator.evaluate_dataset(
        test_samples, 
        vector=vector,
        layer_idx=layer_idx,
        scaling_factor=scaling_factor,
        desc=f"Injection (L{layer_idx})"
    )
