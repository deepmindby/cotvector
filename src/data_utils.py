"""
Data loading and processing utilities for CoT Vectors.
Handles GSM8K, MATH, and MMLU-Pro datasets with proper prompt templates.

Based on Appendix A.2.2 (Table 3) for prompt templates.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


logger = logging.getLogger("cot_vectors")


# ==================== Prompt Templates (Table 3 in Appendix) ====================

# Note: Use double braces {{}} to escape braces that should appear literally in output
# Single braces {question} are format placeholders
PROMPT_TEMPLATES = {
    "gsm8k": {
        "cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Put your answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
    },
    "math": {
        "cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving math problems. "
            "Put your answer within \\boxed{{}}.\n\n"
            "Question: {question}\n"
        ),
    },
    "mmlu_pro": {
        "cot": (
            "You are a helpful and precise assistant for solving problems. "
            "Please reason step by step, and put your final answer within \\boxed{{}}. "
            "Your final output should be only the uppercase letter of the correct choice (e.g., A).\n\n"
            "Question: {question}\n"
            "Choices:\n{choices}\n"
        ),
        "non_cot": (
            "You are a helpful and precise assistant for solving problems. "
            "Put your answer within \\boxed{{}}. "
            "Your final output should be only the uppercase letter of the correct choice (e.g., A).\n\n"
            "Question: {question}\n"
            "Choices:\n{choices}\n"
        ),
    },
}


@dataclass
class CoTSample:
    """
    Data class for a single CoT sample.
    
    Attributes:
        question: The input question
        cot: Chain-of-thought reasoning (optional for non-CoT)
        answer: The final answer
        full_cot_text: Complete formatted text with CoT
        full_non_cot_text: Complete formatted text without CoT
        choices: Multiple choice options (for MMLU-Pro, optional)
    """
    question: str
    cot: Optional[str]
    answer: str
    full_cot_text: str
    full_non_cot_text: str
    choices: Optional[str] = None  # For MMLU-Pro multiple choice
    metadata: Optional[Dict[str, Any]] = None


class CoTDataset(Dataset):
    """
    PyTorch Dataset for CoT Vector training/evaluation.
    """
    
    def __init__(
        self,
        samples: List[CoTSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        include_cot: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            samples: List of CoTSample objects
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            include_cot: Whether to include CoT in tokenization
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_cot = include_cot
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Choose text based on whether to include CoT
        if self.include_cot:
            text = sample.full_cot_text
        else:
            text = sample.full_non_cot_text
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": sample.answer,
            "question": sample.question,
            "cot": sample.cot,
            "full_cot_text": sample.full_cot_text,
            "full_non_cot_text": sample.full_non_cot_text,
        }


def format_cot_sample(
    question: str,
    cot: Optional[str],
    answer: str,
    dataset_type: str,
    choices: Optional[str] = None
) -> CoTSample:
    """
    Format a sample into CoTSample with proper prompt templates.
    
    Args:
        question: The question text
        cot: Chain-of-thought reasoning
        answer: The final answer
        dataset_type: Type of dataset (gsm8k, math, mmlu_pro)
        choices: Multiple choice options (for MMLU-Pro)
    
    Returns:
        CoTSample object
    """
    templates = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    # Format question with choices if applicable
    if dataset_type == "mmlu_pro" and choices:
        question_formatted = question
        cot_prompt = templates["cot"].format(question=question_formatted, choices=choices)
        non_cot_prompt = templates["non_cot"].format(question=question_formatted, choices=choices)
    else:
        cot_prompt = templates["cot"].format(question=question)
        non_cot_prompt = templates["non_cot"].format(question=question)
    
    # Create full text with CoT reasoning and answer
    if cot:
        full_cot_text = f"{cot_prompt}\n{cot}\n\nThe answer is \\boxed{{{answer}}}"
    else:
        full_cot_text = f"{cot_prompt}\nThe answer is \\boxed{{{answer}}}"
    
    # Non-CoT version goes directly to answer
    full_non_cot_text = f"{non_cot_prompt}\nThe answer is \\boxed{{{answer}}}"
    
    return CoTSample(
        question=question,
        cot=cot,
        answer=answer,
        full_cot_text=full_cot_text,
        full_non_cot_text=full_non_cot_text,
        choices=choices if dataset_type == "mmlu_pro" else None,
    )


def load_gsm8k(
    data_path: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """
    Load GSM8K dataset.
    
    Args:
        data_path: Path to the data directory
        split: 'train' or 'test'
        num_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
    
    Returns:
        List of CoTSample objects
    """
    logger.info(f"Loading GSM8K {split} split from {data_path}")
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(data_path, "gsm8k", f"{split}.jsonl"),
        os.path.join(data_path, f"gsm8k_{split}.jsonl"),
        os.path.join(data_path, "gsm8k", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"GSM8K data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease run 'python convert_data.py --data_dir {data_path}' to convert the data first."
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get("question", data.get("problem", ""))
            answer = data.get("answer", "")
            
            # Extract numerical answer from GSM8K format (#### 123)
            if "####" in str(answer):
                parts = str(answer).split("####")
                cot = parts[0].strip() if len(parts) > 1 else None
                final_answer = parts[-1].strip()
            else:
                cot = data.get("cot", data.get("solution", None))
                final_answer = str(answer).strip()
            
            sample = format_cot_sample(
                question=question,
                cot=cot,
                answer=final_answer,
                dataset_type="gsm8k"
            )
            samples.append(sample)
    
    # Subsample if requested
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} GSM8K samples")
    return samples


def load_math(
    data_path: str,
    split: str = "train",
    difficulty: str = "all",  # 'easy', 'hard', or 'all'
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """
    Load MATH dataset.
    
    Args:
        data_path: Path to the data directory
        split: 'train' or 'test'
        difficulty: 'easy' (levels 1-3), 'hard' (levels 4-5), or 'all'
        num_samples: Number of samples to load
        seed: Random seed for sampling
    
    Returns:
        List of CoTSample objects
    """
    logger.info(f"Loading MATH {split} split ({difficulty}) from {data_path}")
    
    # Try different file naming conventions
    possible_paths = [
        os.path.join(data_path, "math", f"{split}.jsonl"),
        os.path.join(data_path, f"math_{split}.jsonl"),
        os.path.join(data_path, "MATH", f"{split}.jsonl"),
        os.path.join(data_path, "math", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    # If test split not found, use train split and take a portion for testing
    use_train_for_test = False
    if file_path is None and split == "test":
        train_paths = [
            os.path.join(data_path, "math", "train.jsonl"),
            os.path.join(data_path, "math_train.jsonl"),
            os.path.join(data_path, "MATH", "train.jsonl"),
        ]
        for path in train_paths:
            if os.path.exists(path):
                file_path = path
                use_train_for_test = True
                logger.warning(f"Test split not found, using portion of train data from {path}")
                break
    
    if file_path is None:
        raise FileNotFoundError(
            f"MATH data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease run 'python convert_data.py --data_dir {data_path}' to convert the data first."
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Filter by difficulty if specified
            level = data.get("level", 3)
            if isinstance(level, str):
                # Parse "Level X" format
                try:
                    level = int(level.replace("Level ", "").strip())
                except ValueError:
                    level = 3
            
            if difficulty == "easy" and level > 3:
                continue
            elif difficulty == "hard" and level < 4:
                continue
            
            # Get fields - handle different naming conventions
            question = data.get("question", data.get("problem", ""))
            solution = data.get("cot", data.get("solution", ""))
            answer = data.get("answer", "")
            
            # Extract answer from solution if not provided
            if not answer and solution:
                import re
                match = re.search(r"\\boxed\{([^}]+)\}", solution)
                if match:
                    answer = match.group(1)
            
            sample = format_cot_sample(
                question=question,
                cot=solution,
                answer=answer,
                dataset_type="math"
            )
            samples.append(sample)
    
    # If using train data for test, take the last 20% as test set
    if use_train_for_test:
        import random
        random.seed(seed)
        random.shuffle(samples)
        test_size = max(len(samples) // 5, 200)  # At least 200 samples or 20%
        samples = samples[-test_size:]
        logger.info(f"Using {len(samples)} samples from train as test set")
    
    # Subsample if requested
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} MATH samples")
    return samples


def load_mmlu_pro(
    data_path: str,
    split: str = "validation",  # Paper uses validation as support set
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """
    Load MMLU-Pro dataset.
    
    Args:
        data_path: Path to the data directory
        split: 'validation' or 'test'
        num_samples: Number of samples to load
        seed: Random seed for sampling
    
    Returns:
        List of CoTSample objects
    """
    logger.info(f"Loading MMLU-Pro {split} split from {data_path}")
    
    possible_paths = [
        os.path.join(data_path, "mmlu_pro", f"{split}.jsonl"),
        os.path.join(data_path, f"mmlu_pro_{split}.jsonl"),
        os.path.join(data_path, "MMLU-Pro", f"{split}.jsonl"),
        os.path.join(data_path, "mmlu_pro", split, "data.jsonl"),
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"MMLU-Pro data file not found. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease run 'python convert_data.py --data_dir {data_path}' to convert the data first."
        )
    
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get("question", "")
            choices = data.get("choices", data.get("options", []))
            answer = data.get("answer", "")
            cot = data.get("cot", data.get("rationale", data.get("explanation", "")))
            
            # Handle answer that might be an index (0, 1, 2, ...) instead of letter (A, B, C, ...)
            if isinstance(answer, int):
                # Convert index to letter
                answer = chr(65 + answer)  # 0->A, 1->B, etc.
            elif isinstance(answer, str) and answer.isdigit():
                # String digit like "0", "1", etc.
                answer = chr(65 + int(answer))
            
            # Format choices
            if isinstance(choices, list):
                choices_text = "\n".join([
                    f"{chr(65+i)}. {choice}" 
                    for i, choice in enumerate(choices)
                ])
            else:
                choices_text = str(choices)
            
            sample = format_cot_sample(
                question=question,
                cot=cot,
                answer=str(answer),
                dataset_type="mmlu_pro",
                choices=choices_text
            )
            samples.append(sample)
    
    # Subsample if requested
    if num_samples and num_samples < len(samples):
        import random
        random.seed(seed)
        samples = random.sample(samples, num_samples)
    
    logger.info(f"Loaded {len(samples)} MMLU-Pro samples")
    return samples


def load_dataset(
    data_path: str,
    dataset_name: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42
) -> List[CoTSample]:
    """
    Load a dataset by name.
    
    Args:
        data_path: Path to the data directory
        dataset_name: Name of the dataset
        split: Data split to load
        num_samples: Number of samples to load
        seed: Random seed
    
    Returns:
        List of CoTSample objects
    """
    loaders = {
        "gsm8k": lambda: load_gsm8k(data_path, split, num_samples, seed),
        "math_easy": lambda: load_math(data_path, split, "easy", num_samples, seed),
        "math_hard": lambda: load_math(data_path, split, "hard", num_samples, seed),
        "math": lambda: load_math(data_path, split, "all", num_samples, seed),
        "mmlu_pro": lambda: load_mmlu_pro(
            data_path, 
            "validation" if split == "train" else split,
            num_samples, 
            seed
        ),
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(loaders.keys())}")
    
    return loaders[dataset_name]()


def create_dataloader(
    samples: List[CoTSample],
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 2048,
    include_cot: bool = True,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader from samples.
    
    Args:
        samples: List of CoTSample objects
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        include_cot: Whether to include CoT in samples
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
    
    Returns:
        PyTorch DataLoader
    """
    dataset = CoTDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_length,
        include_cot=include_cot
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def collate_for_extraction(
    batch: List[Dict],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048
) -> Dict[str, Any]:
    """
    Custom collate function for CoT vector extraction.
    Returns both CoT and non-CoT tokenized versions.
    
    Args:
        batch: List of sample dictionaries
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with CoT and non-CoT batches
    """
    cot_texts = [item["full_cot_text"] for item in batch]
    non_cot_texts = [item["full_non_cot_text"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    # Tokenize CoT versions
    cot_encoding = tokenizer(
        cot_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Tokenize non-CoT versions
    non_cot_encoding = tokenizer(
        non_cot_texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        "cot_input_ids": cot_encoding["input_ids"],
        "cot_attention_mask": cot_encoding["attention_mask"],
        "non_cot_input_ids": non_cot_encoding["input_ids"],
        "non_cot_attention_mask": non_cot_encoding["attention_mask"],
        "answers": answers,
        "cot_texts": cot_texts,
        "non_cot_texts": non_cot_texts,
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing data utilities...")
    
    # Test format function
    sample = format_cot_sample(
        question="What is 2 + 2?",
        cot="Let me think step by step. 2 + 2 equals 4.",
        answer="4",
        dataset_type="gsm8k"
    )
    print(f"Sample question: {sample.question}")
    print(f"Sample answer: {sample.answer}")
    print(f"Full CoT text:\n{sample.full_cot_text[:200]}...")
    
    print("\nData utilities test complete!")
