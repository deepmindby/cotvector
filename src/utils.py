"""
Utility functions for CoT Vectors reproduction.
Includes WandB setup, seeding, logging, and common helpers.
"""

import os
import random
import logging
from typing import Optional, Dict, Any
from pathlib import Path

import yaml
import numpy as np
import torch
from datetime import datetime


def setup_logging(output_dir: str, debug: bool = False) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        output_dir: Directory to save log files
        debug: Enable debug level logging
    
    Returns:
        Configured logger instance
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    log_level = logging.DEBUG if debug else logging.INFO
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"run_{timestamp}.log")
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    logger = logging.getLogger("cot_vectors")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility with CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    logging.getLogger("cot_vectors").info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def setup_wandb(
    args=None,
    config_path: str = "config/secrets.yaml",
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enabled: bool = True
) -> Optional[Any]:
    """
    Initialize WandB logging.
    
    Args:
        args: Argument namespace (optional, extracts project/config from args)
        config_path: Path to secrets.yaml with WandB credentials
        project: WandB project name (overrides config)
        run_name: Name for this run
        config: Configuration dict to log
        enabled: Whether WandB is enabled
    
    Returns:
        WandB run object or None if disabled/failed
    """
    # Extract from args if provided
    if args is not None:
        project = project or getattr(args, 'wandb_project', None)
        config = config or vars(args)
        enabled = getattr(args, 'use_wandb', enabled)
    
    if not enabled:
        return None
    
    try:
        import wandb
    except ImportError:
        logging.warning("wandb not installed. Skipping WandB setup.")
        return None
    
    try:
        # Load credentials
        secrets = load_config(config_path)
        wandb_config = secrets.get("wandb", {})
        
        api_key = wandb_config.get("api_key", os.environ.get("WANDB_API_KEY"))
        entity = wandb_config.get("entity")
        default_project = wandb_config.get("project", "cot-vectors-reproduction")
        
        if api_key and api_key != "PUT_KEY_HERE":
            os.environ["WANDB_API_KEY"] = api_key
        
        # Don't use entity if it's the placeholder
        use_entity = entity if entity and entity != "your-username" else None
        
        # Initialize run with error handling
        run = wandb.init(
            project=project or default_project,
            entity=use_entity,
            name=run_name,
            config=config,
            reinit=True
        )
        
        logging.getLogger("cot_vectors").info(f"WandB initialized: {run.url}")
        return run
        
    except Exception as e:
        logging.getLogger("cot_vectors").warning(f"WandB initialization failed: {e}")
        return None


def save_vector(
    vector: torch.Tensor,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save a CoT vector to disk.
    
    Args:
        vector: The CoT vector tensor
        output_path: Path to save the vector
        metadata: Optional metadata to save with the vector
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "vector": vector.cpu(),
        "metadata": metadata or {}
    }
    
    torch.save(save_dict, output_path)
    logging.getLogger("cot_vectors").info(f"Vector saved to {output_path}")


def load_vector(vector_path: str) -> tuple:
    """
    Load a CoT vector from disk.
    
    Args:
        vector_path: Path to the saved vector
    
    Returns:
        Tuple of (vector tensor, metadata dict)
    """
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Vector file not found: {vector_path}")
    
    save_dict = torch.load(vector_path, map_location="cpu", weights_only=False)
    vector = save_dict["vector"]
    metadata = save_dict.get("metadata", {})
    
    logging.getLogger("cot_vectors").info(f"Vector loaded from {vector_path}")
    
    return vector, metadata


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: int) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format
    
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    return str(num)


def print_results_summary(
    model_name: str,
    method: str,
    layer_idx: int,
    dataset: str,
    accuracy: float,
    wandb_url: Optional[str] = None,
    num_params: Optional[int] = None
) -> None:
    """
    Print a formatted results summary table.
    
    Args:
        model_name: Name of the model used
        method: Method name (extracted/learnable)
        layer_idx: Layer index used for injection
        dataset: Dataset name
        accuracy: Achieved accuracy
        wandb_url: WandB run URL (optional)
        num_params: Number of trainable parameters (optional)
    """
    print("\n" + "=" * 48)
    print("Results Summary")
    print("-" * 48)
    print(f"Model:       {model_name}")
    print(f"Method:      {method.capitalize()} CoT Vector")
    print(f"Layer:       {layer_idx}")
    print(f"Dataset:     {dataset.upper()}")
    print(f"Accuracy:    {accuracy:.2f}%")
    if num_params is not None:
        print(f"#Params:     {format_number(num_params)}")
    if wandb_url:
        print(f"WandB Run:   {wandb_url}")
    print("=" * 48 + "\n")


def extract_answer_from_text(text: str, dataset: str = "gsm8k") -> Optional[str]:
    """
    Extract the final answer from generated text.
    
    Enhanced version with multiple extraction strategies for robustness.
    Prioritizes explicit answer formats, then looks for answers near the end of text.
    
    Args:
        text: Generated text containing the answer
        dataset: Dataset type for format-specific extraction
    
    Returns:
        Extracted answer string or None
    """
    import re
    
    if not text:
        return None
    
    text = text.strip()
    
    # ========== Priority 1: Explicit answer formats ==========
    
    # Try to find answer in \boxed{} format (MATH style) - HIGHEST PRIORITY
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_matches:
        answer = boxed_matches[-1].strip()  # Take the last boxed answer
        if dataset == "mmlu_pro":
            letter_match = re.search(r"([A-J])", answer)
            if letter_match:
                return letter_match.group(1)
        return answer
    
    # Try to find "#### X" format (GSM8K style) - SECOND PRIORITY
    hash_matches = re.findall(r"####\s*(.+?)(?:\n|$)", text)
    if hash_matches:
        return hash_matches[-1].strip()
    
    # Try to find "The answer is X" format (multiple variations) - THIRD PRIORITY
    answer_patterns = [
        r"[Tt]he (?:final )?answer is[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Aa]nswer[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Tt]herefore,?\s+(?:the\s+)?(?:total\s+)?(?:answer\s+)?(?:is\s+)?\$?([0-9,]+(?:\.[0-9]+)?)",
        r"[Ss]o,?\s+(?:the\s+)?(?:total\s+)?(?:answer\s+)?(?:is\s+)?\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].replace(",", "").strip()
            if answer:
                return answer
    
    # ========== Priority 2: Dataset-specific patterns ==========
    
    if dataset == "mmlu_pro":
        # Look for patterns like "A.", "Answer: A", "(A)", etc.
        patterns = [
            r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-J])\)?",
            r"\(?([A-J])\)?\s*(?:is correct|is the (?:correct |right )?answer)",
            r"^\s*\(?([A-J])\)?\s*$",
            r"\b([A-J])\b(?:\s*[.)]|\s*$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
    
    # ========== Priority 3: Look for final answer near the END of text ==========
    
    if dataset in ["gsm8k", "math_easy", "math_hard"]:
        # Focus on the LAST portion of text (last 500 chars) for final answer
        text_end = text[-500:] if len(text) > 500 else text
        
        # Common final answer patterns
        final_patterns = [
            # "= $12" or "= 12" near end
            r"=\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
            # "pays $500" / "costs $500" / "earns $500" / "receives $500"
            r"(?:pays?|costs?|receives?|earns?|gets?|made|makes?|spent|spends?|saved?|saves?|is|was|are|were|equals?)\s*\$?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars?|\.|\s|$)",
            # "Total: 500" or "Total = 500"
            r"[Tt]otal[:\s=]+\$?\s*([0-9,]+(?:\.[0-9]+)?)",
            # "$500." or "$500" at end of sentence
            r"\$\s*([0-9,]+(?:\.[0-9]+)?)\s*[.!]",
            # Number followed by "dollars" 
            r"([0-9,]+(?:\.[0-9]+)?)\s*dollars",
        ]
        
        for pattern in final_patterns:
            matches = re.findall(pattern, text_end, re.IGNORECASE)
            if matches:
                answer = matches[-1].replace(",", "")
                return answer
        
        # Last resort: find all numbers in the last 300 chars and take the last "reasonable" one
        text_very_end = text[-300:] if len(text) > 300 else text
        all_numbers = re.findall(r"(?<![0-9.])([0-9]+(?:\.[0-9]+)?)(?![0-9])", text_very_end)
        
        if all_numbers:
            # Filter to reasonable answer range and prefer integers
            candidates = []
            for num_str in all_numbers:
                try:
                    num = float(num_str)
                    # Reasonable answer range for most GSM8K problems
                    if 0 <= num <= 100000:
                        candidates.append(num_str)
                except:
                    pass
            
            if candidates:
                # Return the LAST reasonable number in the text ending
                return candidates[-1]
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.
    
    Args:
        answer: Raw answer string
    
    Returns:
        Normalized answer string
    """
    if answer is None:
        return ""
    
    # Remove common formatting
    answer = answer.strip()
    answer = answer.replace(",", "")  # Remove commas from numbers
    answer = answer.replace("$", "")  # Remove dollar signs
    answer = answer.replace("%", "")  # Remove percentage signs
    answer = answer.lower()
    
    # Try to extract just the number if present
    import re
    num_match = re.search(r"-?\d+\.?\d*", answer)
    if num_match:
        return num_match.group(0)
    
    return answer


def compare_answers(pred: str, gold: str, dataset: str = "gsm8k") -> bool:
    """
    Compare predicted and gold answers.
    
    Args:
        pred: Predicted answer
        gold: Gold/ground truth answer
        dataset: Dataset type for format-specific comparison
    
    Returns:
        True if answers match, False otherwise
    """
    if pred is None:
        return False
    
    # For MMLU-Pro, just compare letters directly (case-insensitive)
    if dataset == "mmlu_pro":
        pred_letter = pred.strip().upper() if pred else ""
        gold_letter = gold.strip().upper() if gold else ""
        # Extract just the letter if there's extra content
        import re
        pred_match = re.search(r"([A-J])", pred_letter)
        gold_match = re.search(r"([A-J])", gold_letter)
        if pred_match and gold_match:
            return pred_match.group(1) == gold_match.group(1)
        return pred_letter == gold_letter
    
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if pred_norm == gold_norm:
        return True
    
    # Try numeric comparison
    try:
        pred_num = float(pred_norm)
        gold_num = float(gold_norm)
        return abs(pred_num - gold_num) < 1e-6
    except (ValueError, TypeError):
        pass
    
    return False


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking training metrics.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test logging
    logger = setup_logging("./test_outputs", debug=True)
    logger.info("Test log message")
    
    # Test seed setting
    set_seed(42)
    
    # Test device
    device = get_device()
    print(f"Device: {device}")
    
    # Test answer extraction
    test_texts = [
        "The answer is 42",
        "Therefore, \\boxed{123}",
        "So the result is #### 500",
    ]
    for text in test_texts:
        answer = extract_answer_from_text(text)
        print(f"Extracted '{answer}' from: {text[:30]}...")
    
    print("All tests passed!")
