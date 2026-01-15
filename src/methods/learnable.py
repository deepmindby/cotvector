"""
Learnable CoT Vector implementation.
Implements the teacher-student framework from Section 3.2.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import math
import gc

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES


class CoTDataset(Dataset):
    """Dataset for CoT vector training."""
    
    def __init__(self, samples: List, tokenizer, dataset_type: str, max_length: int = 1024):
        self.samples = samples
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type
        self.max_length = max_length
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Build prompts
        if self.dataset_type == "mmlu_pro":
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question,
                choices=sample.choices
            ) + f"The answer is {sample.answer}"
        else:
            teacher_prompt = self.prompt_template["cot"].format(
                question=sample.question
            ) + sample.cot + f"\nThe answer is {sample.answer}"
            
            student_prompt = self.prompt_template["non_cot"].format(
                question=sample.question
            ) + f"The answer is {sample.answer}"
        
        # Tokenize without padding (will pad in collate_fn)
        teacher_enc = self.tokenizer(
            teacher_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
        )
        student_enc = self.tokenizer(
            student_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        
        # Get answer token positions
        answer_text = f"The answer is {sample.answer}"
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        answer_len = len(answer_ids)
        
        # Actual sequence lengths
        teacher_len = teacher_enc["input_ids"].shape[1]
        student_len = student_enc["input_ids"].shape[1]
        
        return {
            "teacher_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_mask": teacher_enc["attention_mask"].squeeze(0),
            "student_ids": student_enc["input_ids"].squeeze(0),
            "student_mask": student_enc["attention_mask"].squeeze(0),
            "teacher_len": teacher_len,
            "student_len": student_len,
            "answer_len": answer_len,
        }


def collate_fn(batch):
    """Custom collate function with dynamic padding."""
    # Find max lengths in this batch
    max_teacher_len = max(item["teacher_len"] for item in batch)
    max_student_len = max(item["student_len"] for item in batch)
    
    teacher_ids_list = []
    teacher_mask_list = []
    student_ids_list = []
    student_mask_list = []
    teacher_lens = []
    student_lens = []
    answer_lens = []
    
    for item in batch:
        # Pad teacher
        t_ids = item["teacher_ids"]
        t_mask = item["teacher_mask"]
        t_pad_len = max_teacher_len - len(t_ids)
        if t_pad_len > 0:
            t_ids = F.pad(t_ids, (0, t_pad_len), value=0)
            t_mask = F.pad(t_mask, (0, t_pad_len), value=0)
        teacher_ids_list.append(t_ids)
        teacher_mask_list.append(t_mask)
        
        # Pad student
        s_ids = item["student_ids"]
        s_mask = item["student_mask"]
        s_pad_len = max_student_len - len(s_ids)
        if s_pad_len > 0:
            s_ids = F.pad(s_ids, (0, s_pad_len), value=0)
            s_mask = F.pad(s_mask, (0, s_pad_len), value=0)
        student_ids_list.append(s_ids)
        student_mask_list.append(s_mask)
        
        teacher_lens.append(item["teacher_len"])
        student_lens.append(item["student_len"])
        answer_lens.append(item["answer_len"])
    
    return {
        "teacher_ids": torch.stack(teacher_ids_list),
        "teacher_mask": torch.stack(teacher_mask_list),
        "student_ids": torch.stack(student_ids_list),
        "student_mask": torch.stack(student_mask_list),
        "teacher_len": teacher_lens,
        "student_len": student_lens,
        "answer_len": answer_lens,
    }


class LearnableCoTVector(BaseCoTVectorMethod):
    """
    Learnable CoT Vector optimized via teacher-student framework.
    
    Loss = L_align + λ * L_CE
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        lambda_val: float = 0.5,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-3,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 2,  # Reduced default batch size
        gradient_accumulation_steps: int = 4,  # Increased accumulation
        max_length: int = 1024,  # Reduced max length
    ):
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.lambda_val = lambda_val
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        
        # Initialize learnable vector
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.vector_param, std=0.02)
    
    def _compute_alignment_loss(
        self,
        teacher_hidden: torch.Tensor,
        student_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE alignment loss (more memory efficient than KL)."""
        # Simple MSE loss for alignment
        loss = F.mse_loss(student_hidden, teacher_hidden.detach())
        return loss
    
    def _compute_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss on answer tokens."""
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        
        # Flatten
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = shift_mask.view(-1).float()
        
        # Compute loss only on masked positions
        ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        masked_loss = (ce_loss * flat_mask).sum() / (flat_mask.sum() + 1e-8)
        
        return masked_loss
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """Train the learnable CoT vector."""
        print(f"Training learnable vector at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Epochs: {self.num_epochs}")
        print(f"  LR: {self.learning_rate}, λ: {self.lambda_val}")
        print(f"  Batch size: {self.batch_size}, Grad accum: {self.gradient_accumulation_steps}")
        print(f"  Max length: {self.max_length}")
        
        # Create dataset and dataloader
        dataset = CoTDataset(
            support_samples, 
            self.tokenizer, 
            self.dataset_type,
            max_length=self.max_length
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=False,  # Disable pin_memory to reduce memory
            collate_fn=collate_fn,
        )
        
        # Get target device for vector
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        target_device = next(target_layer.parameters()).device
        self.vector_param.data = self.vector_param.data.to(target_device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        # Learning rate scheduler
        total_steps = len(dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        global_step = 0
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_align = 0.0
            epoch_ce = 0.0
            num_batches = 0
            
            optimizer.zero_grad()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", ncols=100)
            
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move to device
                    teacher_ids = batch["teacher_ids"].to(target_device)
                    teacher_mask = batch["teacher_mask"].to(target_device)
                    student_ids = batch["student_ids"].to(target_device)
                    student_mask = batch["student_mask"].to(target_device)
                    
                    bs = teacher_ids.size(0)
                    
                    # ========== Teacher forward (frozen, no grad) ==========
                    self.model_wrapper.clear_hooks()
                    self.model_wrapper.register_extraction_hook(
                        self.layer_idx, 
                        requires_grad=False  # No grad for teacher
                    )
                    
                    with torch.no_grad():
                        self.model_wrapper(teacher_ids, attention_mask=teacher_mask)
                    
                    # Get teacher hidden states and immediately detach/clone
                    teacher_hidden_raw = self.model_wrapper.get_activations(self.layer_idx)
                    
                    # Extract answer positions for teacher
                    teacher_answer_hiddens = []
                    for i in range(bs):
                        t_len = batch["teacher_len"][i]
                        a_len = batch["answer_len"][i]
                        t_ans_pos = list(range(max(0, t_len - a_len), t_len))
                        if t_ans_pos:
                            h = teacher_hidden_raw[i, t_ans_pos, :].mean(dim=0)
                            teacher_answer_hiddens.append(h)
                    
                    # Clear teacher activations to free memory
                    self.model_wrapper.clear_hooks()
                    del teacher_hidden_raw
                    
                    if not teacher_answer_hiddens:
                        continue
                    
                    teacher_hidden = torch.stack(teacher_answer_hiddens)  # [valid_bs, hidden]
                    del teacher_answer_hiddens
                    
                    # ========== Student forward (with injection, need grad) ==========
                    self.model_wrapper.register_injection_hook(
                        self.layer_idx, 
                        self.vector_param,
                        scaling_factor=1.0,
                        requires_grad=True  # Need grad for training
                    )
                    self.model_wrapper.register_extraction_hook(
                        self.layer_idx,
                        requires_grad=True  # Need grad for alignment loss
                    )
                    
                    student_outputs = self.model_wrapper(student_ids, attention_mask=student_mask)
                    student_hidden_raw = self.model_wrapper.get_activations(self.layer_idx)
                    student_logits = student_outputs.logits
                    
                    # Extract answer positions for student
                    student_answer_hiddens = []
                    ce_losses = []
                    valid_indices = []
                    
                    for i in range(bs):
                        s_len = batch["student_len"][i]
                        a_len = batch["answer_len"][i]
                        s_ans_pos = list(range(max(0, s_len - a_len), s_len))
                        
                        if s_ans_pos and i < len(teacher_hidden):
                            h = student_hidden_raw[i, s_ans_pos, :].mean(dim=0)
                            student_answer_hiddens.append(h)
                            valid_indices.append(i)
                            
                            # CE loss for this sample
                            ans_mask = torch.zeros(student_mask.shape[1], device=target_device)
                            ans_mask[s_ans_pos] = 1
                            ce_loss = self._compute_ce_loss(
                                student_logits[i:i+1],
                                student_ids[i:i+1],
                                ans_mask.unsqueeze(0)
                            )
                            ce_losses.append(ce_loss)
                    
                    if not student_answer_hiddens:
                        self.model_wrapper.clear_hooks()
                        continue
                    
                    student_hidden = torch.stack(student_answer_hiddens)  # [valid_bs, hidden]
                    
                    # Filter teacher hidden to match valid indices
                    teacher_hidden_filtered = teacher_hidden[:len(student_hidden)]
                    
                    # ========== Compute losses ==========
                    # Alignment loss
                    align_loss = self._compute_alignment_loss(teacher_hidden_filtered, student_hidden)
                    
                    # CE loss
                    ce_loss = torch.stack(ce_losses).mean() if ce_losses else torch.tensor(0.0, device=target_device)
                    
                    # Combined loss
                    loss = align_loss + self.lambda_val * ce_loss
                    loss = loss / self.gradient_accumulation_steps
                    
                    # Backward
                    loss.backward()
                    
                    # Clear hooks and intermediate tensors
                    self.model_wrapper.clear_hooks()
                    del student_hidden_raw, student_logits, student_outputs
                    del teacher_hidden, student_hidden, teacher_hidden_filtered
                    del student_answer_hiddens, ce_losses
                    
                    # Gradient accumulation step
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_([self.vector_param], 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Clear CUDA cache periodically
                        if global_step % 10 == 0:
                            torch.cuda.empty_cache()
                    
                    # Track losses
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    epoch_align += align_loss.item()
                    epoch_ce += ce_loss.item()
                    num_batches += 1
                    
                    # Update progress
                    pbar.set_postfix({
                        "loss": f"{epoch_loss/num_batches:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                    })
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Handle OOM gracefully
                        print(f"\n  Warning: OOM at batch {batch_idx}, clearing cache and skipping...")
                        self.model_wrapper.clear_hooks()
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise
            
            # End of epoch: ensure optimizer step for remaining gradients
            if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_([self.vector_param], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            avg_align = epoch_align / max(num_batches, 1)
            avg_ce = epoch_ce / max(num_batches, 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, align={avg_align:.4f}, ce={avg_ce:.4f}")
            
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/align_loss": avg_align,
                    "train/ce_loss": avg_ce,
                    "train/lr": scheduler.get_last_lr()[0],
                })
            
            # Track best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.vector = self.vector_param.detach().clone()
            
            # Clear cache at end of epoch
            torch.cuda.empty_cache()
            gc.collect()
        
        # Final vector
        if self.vector is None:
            self.vector = self.vector_param.detach().clone()
        
        print(f"Training complete. Vector norm: {self.vector.norm().item():.4f}")
        
        return self.vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        return self.vector
