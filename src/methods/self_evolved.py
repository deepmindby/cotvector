"""
Self-Evolved CoT Vector via GRPO (Group Relative Policy Optimization).

This module implements a novel approach where the CoT Vector evolves through
reinforcement learning, exploring reasoning paths and receiving binary rewards
based on final answer correctness.

Algorithm based on the GRPO framework:
1. For each question, generate G outputs with the current vector
2. Compute binary rewards (correct=1, incorrect=0)
3. Calculate group-relative advantages
4. Update vector using policy gradient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Tuple
from tqdm import tqdm
import random

from .base import BaseCoTVectorMethod
from ..models import CoTModelWrapper
from ..data_utils import PROMPT_TEMPLATES
from ..utils import extract_answer_from_text, compare_answers


class SelfEvolvedCoTVector(BaseCoTVectorMethod):
    """
    Self-Evolved CoT Vector optimized via Group Relative Policy Optimization (GRPO).
    
    Instead of supervised learning from a teacher, this method lets the vector
    evolve by exploring reasoning paths and receiving binary rewards based on
    the final answer correctness.
    
    Key differences from Learnable CoT Vector:
    - No teacher model required
    - Uses RL-based optimization (GRPO)
    - Binary reward signal (correct/incorrect)
    - Group-relative advantage estimation
    """
    
    def __init__(
        self,
        model_wrapper: CoTModelWrapper,
        tokenizer,
        layer_idx: int,
        dataset_type: str = "gsm8k",
        group_size: int = 4,
        num_iterations: int = 100,
        learning_rate: float = 1e-3,
        beta: float = 0.01,
        epsilon: float = 1e-8,
        max_new_tokens: int = 512,
        questions_per_iter: int = 2,
        temperature: float = 0.7,
    ):
        """
        Initialize Self-Evolved CoT Vector.
        
        Args:
            model_wrapper: Wrapper around the LLM
            tokenizer: Tokenizer for the model
            layer_idx: Layer index for vector injection
            dataset_type: Type of dataset (gsm8k, math, mmlu_pro)
            group_size: Number of samples generated per question (G)
            num_iterations: Number of training iterations
            learning_rate: Learning rate for vector optimization
            beta: KL penalty coefficient (reserved for stability)
            epsilon: Small value for numerical stability
            max_new_tokens: Maximum tokens to generate
            questions_per_iter: Number of questions per iteration
            temperature: Sampling temperature for exploration
        """
        super().__init__(model_wrapper, tokenizer, layer_idx, dataset_type)
        
        self.group_size = group_size
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.max_new_tokens = max_new_tokens
        self.questions_per_iter = questions_per_iter
        self.temperature = temperature
        
        # Get prompt template
        self.prompt_template = PROMPT_TEMPLATES.get(dataset_type, PROMPT_TEMPLATES["gsm8k"])
        
        # Initialize learnable vector parameter
        hidden_size = model_wrapper.hidden_size
        self.vector_param = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.vector_param, std=0.02)
        
        # Get target device for the vector (same as target layer)
        target_layer = self.model_wrapper._get_layer(self.layer_idx)
        self.target_device = next(target_layer.parameters()).device
        self.target_dtype = next(target_layer.parameters()).dtype
        
        # Move vector to target device
        self.vector_param.data = self.vector_param.data.to(
            device=self.target_device,
            dtype=torch.float32  # Keep in float32 for gradient stability
        )
    
    def _build_prompt(self, sample) -> str:
        """Build the CoT prompt for a sample."""
        if self.dataset_type == "mmlu_pro":
            prompt = self.prompt_template["cot"].format(
                question=sample.question,
                choices=sample.choices
            )
        else:
            prompt = self.prompt_template["cot"].format(question=sample.question)
        return prompt
    
    def _generate_samples(
        self,
        prompt: str,
        num_samples: int,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor]]:
        """
        Generate multiple samples for a given prompt with vector injection.
        
        Returns:
            List of tuples: (generated_text, input_ids, generated_ids)
        """
        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        input_ids = encoding["input_ids"].to(self.target_device)
        attention_mask = encoding["attention_mask"].to(self.target_device)
        input_len = input_ids.shape[1]
        
        # Clear hooks and register injection hook
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_injection_hook(
            self.layer_idx,
            self.vector_param,
            scaling_factor=1.0,
            requires_grad=False  # No grad during generation
        )
        
        results = []
        
        # Generate samples one by one (for diversity via sampling)
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.model_wrapper.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_ids = outputs[0, input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            results.append((generated_text, input_ids.clone(), generated_ids.clone()))
        
        self.model_wrapper.clear_hooks()
        return results
    
    def _compute_rewards(
        self,
        generated_texts: List[str],
        ground_truth: str,
    ) -> torch.Tensor:
        """
        Compute binary rewards for generated samples.
        
        Args:
            generated_texts: List of generated reasoning texts
            ground_truth: Ground truth answer
            
        Returns:
            Tensor of rewards (1 for correct, 0 for incorrect)
        """
        rewards = []
        for text in generated_texts:
            predicted = extract_answer_from_text(text, self.dataset_type)
            is_correct = compare_answers(predicted, ground_truth, self.dataset_type)
            rewards.append(1.0 if is_correct else 0.0)
        
        return torch.tensor(rewards, device=self.target_device, dtype=torch.float32)
    
    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        A_i = (r_i - mean(r)) / (std(r) + epsilon)
        
        Args:
            rewards: Tensor of rewards for the group
            
        Returns:
            Tensor of normalized advantages
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        # Normalize advantages
        advantages = (rewards - mean_reward) / (std_reward + self.epsilon)
        
        return advantages
    
    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities of generated tokens with vector injection.
        
        This requires a forward pass with gradient tracking.
        
        Args:
            input_ids: Input token IDs (prompt)
            generated_ids: Generated token IDs
            
        Returns:
            Sum of log probabilities for the generated sequence
        """
        # Concatenate input and generated for full sequence
        full_ids = torch.cat([input_ids.squeeze(0), generated_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        
        # Clear hooks and register injection hook with gradient
        self.model_wrapper.clear_hooks()
        self.model_wrapper.register_injection_hook(
            self.layer_idx,
            self.vector_param,
            scaling_factor=1.0,
            requires_grad=True
        )
        
        # Forward pass
        outputs = self.model_wrapper(full_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute log probs only for generated tokens
        input_len = input_ids.shape[1]
        gen_len = generated_ids.shape[0]
        
        # Shift logits and labels for causal LM
        # logits[:, input_len-1:-1] predicts tokens at positions [input_len, end]
        shift_logits = logits[:, input_len-1:-1, :].contiguous()
        shift_labels = full_ids[:, input_len:].contiguous()
        
        # Compute log softmax
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Sum log probs for the sequence
        total_log_prob = token_log_probs.sum()
        
        self.model_wrapper.clear_hooks()
        
        return total_log_prob
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """
        Train the Self-Evolved CoT Vector using GRPO.
        
        Algorithm:
        1. Sample questions from dataset
        2. For each question, generate G outputs with current vector
        3. Compute binary rewards (correct=1, incorrect=0)
        4. Compute group-relative advantages
        5. Update vector using policy gradient: L = -A * log_prob
        
        Args:
            support_samples: List of training samples
            wandb_run: Optional WandB run for logging
            
        Returns:
            Trained CoT vector
        """
        print(f"Training Self-Evolved CoT Vector via GRPO at layer {self.layer_idx}...")
        print(f"  Samples: {len(support_samples)}, Iterations: {self.num_iterations}")
        print(f"  Group size: {self.group_size}, Questions/iter: {self.questions_per_iter}")
        print(f"  LR: {self.learning_rate}, Temperature: {self.temperature}")
        
        # Setup optimizer (only optimizing the vector)
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        
        # Training statistics
        total_rewards = []
        total_losses = []
        best_avg_reward = 0.0
        best_vector = None
        
        # Main training loop
        pbar = tqdm(range(self.num_iterations), desc="GRPO Training", ncols=100)
        
        for iteration in pbar:
            iter_rewards = []
            iter_losses = []
            
            # Sample questions for this iteration
            sampled_questions = random.sample(
                support_samples,
                min(self.questions_per_iter, len(support_samples))
            )
            
            for sample in sampled_questions:
                # Build prompt
                prompt = self._build_prompt(sample)
                ground_truth = sample.answer
                
                # ==================== Step 1: Exploration via Sampling ====================
                # Generate G outputs for this question
                generated_samples = self._generate_samples(prompt, self.group_size)
                generated_texts = [s[0] for s in generated_samples]
                
                # ==================== Step 2: Reward Calculation ====================
                rewards = self._compute_rewards(generated_texts, ground_truth)
                iter_rewards.extend(rewards.tolist())
                
                # Skip if all rewards are the same (no gradient signal)
                if rewards.std() < self.epsilon:
                    continue
                
                # ==================== Step 3: Advantage Estimation ====================
                advantages = self._compute_advantages(rewards)
                
                # ==================== Step 4: Gradient Update ====================
                # Compute policy gradient loss: L = -sum(A_i * log_prob_i)
                loss = torch.tensor(0.0, device=self.target_device, requires_grad=True)
                
                for i, (text, input_ids, generated_ids) in enumerate(generated_samples):
                    # Skip samples with zero advantage (no contribution to gradient)
                    if abs(advantages[i].item()) < self.epsilon:
                        continue
                    
                    # Compute log probability with gradient
                    log_prob = self._compute_log_probs(input_ids, generated_ids)
                    
                    # Policy gradient: -advantage * log_prob
                    sample_loss = -advantages[i] * log_prob
                    loss = loss + sample_loss
                
                # Normalize loss by group size
                loss = loss / self.group_size
                
                if loss.requires_grad:
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_([self.vector_param], max_norm=1.0)
                    
                    optimizer.step()
                    
                    iter_losses.append(loss.item())
            
            # Compute iteration statistics
            avg_reward = sum(iter_rewards) / max(len(iter_rewards), 1)
            avg_loss = sum(iter_losses) / max(len(iter_losses), 1) if iter_losses else 0.0
            
            total_rewards.append(avg_reward)
            total_losses.append(avg_loss)
            
            # Track best vector
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_vector = self.vector_param.detach().clone()
            
            # Update progress bar
            pbar.set_postfix({
                "reward": f"{avg_reward:.3f}",
                "loss": f"{avg_loss:.4f}",
                "best": f"{best_avg_reward:.3f}"
            })
            
            # WandB logging
            if wandb_run and iteration % 10 == 0:
                wandb_run.log({
                    "iteration": iteration,
                    "train/avg_reward": avg_reward,
                    "train/loss": avg_loss,
                    "train/best_reward": best_avg_reward,
                    "train/vector_norm": self.vector_param.norm().item(),
                })
            
            # Print periodic summary
            if (iteration + 1) % 20 == 0:
                print(f"\n  Iter {iteration+1}: avg_reward={avg_reward:.3f}, "
                      f"loss={avg_loss:.4f}, best={best_avg_reward:.3f}")
        
        # Use best vector or current if no improvement found
        if best_vector is not None:
            self.vector = best_vector
        else:
            self.vector = self.vector_param.detach().clone()
        
        # Final summary
        final_avg_reward = sum(total_rewards[-10:]) / min(10, len(total_rewards))
        print(f"\nTraining complete!")
        print(f"  Final avg reward (last 10 iters): {final_avg_reward:.3f}")
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Vector norm: {self.vector.norm().item():.4f}")
        
        return self.vector
    
    def get_vector(self) -> Optional[torch.Tensor]:
        """Get the trained CoT vector."""
        return self.vector


class SelfEvolvedCoTVectorV2(SelfEvolvedCoTVector):
    """
    Enhanced version with batch processing for better GPU utilization.
    
    This version processes multiple samples in parallel when possible,
    which is more efficient on high-memory GPUs like H100.
    """
    
    def _generate_samples_batch(
        self,
        prompts: List[str],
        num_samples_per_prompt: int,
    ) -> Dict[int, List[Tuple[str, torch.Tensor, torch.Tensor]]]:
        """
        Generate samples for multiple prompts in a batched manner.
        
        Note: Due to variable length generation, we still generate one at a time
        but this method provides a cleaner interface for batch processing.
        """
        results = {}
        
        for idx, prompt in enumerate(prompts):
            results[idx] = self._generate_samples(prompt, num_samples_per_prompt)
        
        return results
    
    def train(
        self,
        support_samples: List,
        wandb_run=None,
    ) -> torch.Tensor:
        """
        Enhanced training with better memory management and logging.
        """
        print(f"Training Self-Evolved CoT Vector V2 via GRPO at layer {self.layer_idx}...")
        print(f"  Total samples: {len(support_samples)}")
        print(f"  Iterations: {self.num_iterations}")
        print(f"  Group size G: {self.group_size}")
        print(f"  Questions per iteration: {self.questions_per_iter}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Temperature: {self.temperature}")
        print("-" * 50)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            [self.vector_param],
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        
        # Learning rate scheduler (cosine annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_iterations,
            eta_min=self.learning_rate * 0.1
        )
        
        # Statistics tracking
        reward_history = []
        loss_history = []
        correct_count = 0
        total_count = 0
        best_avg_reward = 0.0
        best_vector = None
        
        pbar = tqdm(range(self.num_iterations), desc="GRPO Training", ncols=120)
        
        for iteration in pbar:
            iter_rewards = []
            iter_losses = []
            
            # Sample questions
            batch_samples = random.sample(
                support_samples,
                min(self.questions_per_iter, len(support_samples))
            )
            
            accumulated_loss = torch.tensor(0.0, device=self.target_device)
            valid_updates = 0
            
            for sample in batch_samples:
                prompt = self._build_prompt(sample)
                ground_truth = sample.answer
                
                # Step 1: Exploration
                try:
                    generated_samples = self._generate_samples(prompt, self.group_size)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print(f"\nWarning: OOM at iteration {iteration}, skipping sample")
                        continue
                    raise
                
                generated_texts = [s[0] for s in generated_samples]
                
                # Step 2: Rewards
                rewards = self._compute_rewards(generated_texts, ground_truth)
                iter_rewards.extend(rewards.tolist())
                
                # Track accuracy
                correct_count += rewards.sum().item()
                total_count += len(rewards)
                
                # Step 3: Advantages
                if rewards.std() < self.epsilon:
                    # All same reward, skip this sample
                    continue
                
                advantages = self._compute_advantages(rewards)
                
                # Step 4: Policy gradient
                sample_loss = torch.tensor(0.0, device=self.target_device)
                
                for i, (text, input_ids, generated_ids) in enumerate(generated_samples):
                    if abs(advantages[i].item()) < self.epsilon:
                        continue
                    
                    try:
                        log_prob = self._compute_log_probs(input_ids, generated_ids)
                        sample_loss = sample_loss - advantages[i] * log_prob
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            continue
                        raise
                
                sample_loss = sample_loss / self.group_size
                accumulated_loss = accumulated_loss + sample_loss
                valid_updates += 1
            
            # Update if we have valid gradients
            if valid_updates > 0 and accumulated_loss.requires_grad:
                accumulated_loss = accumulated_loss / valid_updates
                
                optimizer.zero_grad()
                accumulated_loss.backward()
                torch.nn.utils.clip_grad_norm_([self.vector_param], max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                iter_losses.append(accumulated_loss.item())
            
            # Statistics
            avg_reward = sum(iter_rewards) / max(len(iter_rewards), 1)
            avg_loss = sum(iter_losses) / max(len(iter_losses), 1) if iter_losses else 0.0
            
            reward_history.append(avg_reward)
            loss_history.append(avg_loss)
            
            # Track best
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_vector = self.vector_param.detach().clone()
            
            # Update progress bar
            accuracy = correct_count / max(total_count, 1) * 100
            pbar.set_postfix({
                "R": f"{avg_reward:.2f}",
                "L": f"{avg_loss:.3f}",
                "Acc": f"{accuracy:.1f}%",
                "Best": f"{best_avg_reward:.2f}"
            })
            
            # WandB logging
            if wandb_run and iteration % 5 == 0:
                wandb_run.log({
                    "iteration": iteration,
                    "train/reward": avg_reward,
                    "train/loss": avg_loss,
                    "train/accuracy": accuracy,
                    "train/best_reward": best_avg_reward,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/vector_norm": self.vector_param.norm().item(),
                })
        
        # Finalize
        self.vector = best_vector if best_vector is not None else self.vector_param.detach().clone()
        
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"  Best avg reward: {best_avg_reward:.3f}")
        print(f"  Final accuracy: {correct_count/max(total_count,1)*100:.1f}%")
        print(f"  Vector norm: {self.vector.norm().item():.4f}")
        print("=" * 50)
        
        return self.vector
