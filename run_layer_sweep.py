#!/usr/bin/env python3
"""
Layer sweep script for CoT Vectors.
Evaluates injection at different layers to find optimal performance.

Supports all methods: extracted, learnable, self_evolved (GRPO)
"""

import argparse
import os
import torch
from datetime import datetime
from tqdm import tqdm

from src.models import CoTModelWrapper, load_tokenizer
from src.data_utils import load_dataset
from src.methods.extracted import ExtractedCoTVector
from src.methods.learnable import LearnableCoTVector
from src.methods.self_evolved import SelfEvolvedCoTVector, SelfEvolvedCoTVectorV2
from src.eval import run_baseline_evaluation, run_injection_evaluation
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="Layer sweep for CoT vectors")
    
    # ==================== Model & Data ====================
    parser.add_argument("--model_path", type=str, default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["qwen", "llama"])
    parser.add_argument("--data_path", type=str, default="/home/haichao/TA/CoTVRL/data")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math_easy", "math_hard", "mmlu_pro"])
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--num_support_samples", type=int, default=1000)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    # ==================== Method Selection ====================
    parser.add_argument("--method", type=str, default="extracted",
                        choices=["extracted", "learnable", "self_evolved"],
                        help="CoT Vector acquisition method")
    
    # ==================== Layer Selection ====================
    parser.add_argument("--layers", type=str, default=None, 
                        help="Comma-separated layers to test (e.g., '0,5,10'). Default: all layers")
    parser.add_argument("--layer_step", type=int, default=2,
                        help="Step size when testing all layers (e.g., 2 = test every 2nd layer)")
    
    # ==================== Generation Config ====================
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--use_early_stopping", action="store_true", default=False)
    
    # ==================== Evaluation Options ====================
    parser.add_argument("--skip_baseline", action="store_true", default=False,
                        help="Skip baseline evaluation, only evaluate injection")
    parser.add_argument("--baseline_accuracy", type=float, default=90.0,
                        help="Pre-computed baseline accuracy (use with --skip_baseline)")
    
    # ==================== Learnable Vector Config ====================
    parser.add_argument("--lambda_val", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)  # Reduced for memory
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # Increased
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max sequence length for learnable method")
    
    # ==================== Self-Evolved (GRPO) Config ====================
    parser.add_argument("--group_size", type=int, default=4,
                        help="Number of samples generated per question (G)")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of GRPO training iterations per layer")
    parser.add_argument("--grpo_lr", type=float, default=1e-3,
                        help="Learning rate for GRPO")
    parser.add_argument("--questions_per_iter", type=int, default=2,
                        help="Questions sampled per iteration")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for GRPO sampling")
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    
    # ==================== Saving Options ====================
    parser.add_argument("--save_vectors", action="store_true", default=False,
                        help="Save vectors for each layer")
    parser.add_argument("--load_vectors_dir", type=str, default=None,
                        help="Load pre-trained vectors from directory (skip training)")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 70)
    print("CoT Vector Layer Sweep")
    print("=" * 70)
    print(f"Model:    {args.model_path.split('/')[-1]}")
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    print(f"Support:  {args.num_support_samples}, Test: {args.num_test_samples}")
    print(f"Skip baseline: {args.skip_baseline}")
    if args.method == "learnable":
        print(f"Learnable Config: epochs={args.num_epochs}, batch={args.batch_size}, "
              f"lr={args.learning_rate}, λ={args.lambda_val}, max_len={args.max_length}")
    if args.method == "self_evolved":
        print(f"GRPO Config: G={args.group_size}, iters={args.num_iterations}, "
              f"Q/iter={args.questions_per_iter}, lr={args.grpo_lr}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    tokenizer = load_tokenizer(args.model_path)
    num_layers = model_wrapper.num_layers
    print(f"Model has {num_layers} layers")
    
    # Load data
    print("\nLoading data...")
    support_samples = load_dataset(args.data_path, args.dataset, "train", args.num_support_samples)
    test_samples = load_dataset(args.data_path, args.dataset, "test", args.num_test_samples)
    print(f"Support: {len(support_samples)}, Test: {len(test_samples)}")
    
    # Determine layers to test
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    else:
        layers = list(range(0, num_layers, args.layer_step))
    
    print(f"\nLayers to test: {layers}")
    print(f"Total: {len(layers)} layers")
    
    # Baseline evaluation (optional)
    baseline_accuracy = args.baseline_accuracy
    
    if not args.skip_baseline:
        print("\n" + "-" * 70)
        print("Running baseline evaluation...")
        baseline = run_baseline_evaluation(
            model_wrapper, tokenizer, test_samples, args.dataset,
            args.max_new_tokens, args.num_beams, args.use_early_stopping
        )
        baseline_accuracy = baseline['accuracy']
        print(f"Baseline accuracy: {baseline_accuracy:.2f}%")
    elif baseline_accuracy is not None:
        print(f"\nUsing provided baseline accuracy: {baseline_accuracy:.2f}%")
    else:
        print("\nSkipping baseline evaluation (no baseline_accuracy provided)")
        baseline_accuracy = 0.0  # Placeholder
    
    # Layer sweep
    print("\n" + "-" * 70)
    print(f"Testing {len(layers)} layers with method: {args.method}")
    print("-" * 70)
    
    results = []
    vectors_dict = {}
    
    for layer_idx in tqdm(layers, desc="Layer sweep", ncols=100):
        print(f"\n>>> Layer {layer_idx}")
        
        vector = None
        
        # Check if we should load a pre-trained vector
        if args.load_vectors_dir:
            vector_path = os.path.join(
                args.load_vectors_dir,
                f"{args.method}_{args.dataset}_L{layer_idx}.pt"
            )
            if os.path.exists(vector_path):
                print(f"  Loading vector from {vector_path}")
                vector = torch.load(vector_path)
            else:
                print(f"  Vector not found at {vector_path}, training new one...")
        
        # Train/Extract vector if not loaded
        if vector is None:
            try:
                if args.method == "extracted":
                    method = ExtractedCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                    )
                    vector = method.extract(support_samples)
                    
                elif args.method == "learnable":
                    method = LearnableCoTVector(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        lambda_val=args.lambda_val,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        warmup_ratio=args.warmup_ratio,
                        num_epochs=args.num_epochs,
                        batch_size=args.batch_size,
                        gradient_accumulation_steps=args.gradient_accumulation_steps,
                        max_length=args.max_length,
                    )
                    vector = method.train(support_samples)
                    
                elif args.method == "self_evolved":
                    method = SelfEvolvedCoTVectorV2(
                        model_wrapper=model_wrapper,
                        tokenizer=tokenizer,
                        layer_idx=layer_idx,
                        dataset_type=args.dataset,
                        group_size=args.group_size,
                        num_iterations=args.num_iterations,
                        learning_rate=args.grpo_lr,
                        beta=args.beta,
                        epsilon=args.epsilon,
                        max_new_tokens=args.max_new_tokens,
                        questions_per_iter=args.questions_per_iter,
                        temperature=args.temperature,
                    )
                    vector = method.train(support_samples)
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print(f"  Layer {layer_idx}: CUDA OOM - {e}")
                    torch.cuda.empty_cache()
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': f"CUDA OOM: {str(e)[:100]}",
                    })
                    continue
                else:
                    print(f"  Layer {layer_idx}: Training Error - {e}")
                    results.append({
                        'layer': layer_idx,
                        'accuracy': 0,
                        'diff': -baseline_accuracy if baseline_accuracy else 0,
                        'error': str(e),
                    })
                    continue
            except Exception as e:
                print(f"  Layer {layer_idx}: Training Error - {e}")
                results.append({
                    'layer': layer_idx,
                    'accuracy': 0,
                    'diff': -baseline_accuracy if baseline_accuracy else 0,
                    'error': str(e),
                })
                continue
        
        # Save vector if requested
        if args.save_vectors and vector is not None:
            vector_path = os.path.join(
                args.output_dir,
                f"{args.method}_{args.dataset}_L{layer_idx}.pt"
            )
            torch.save(vector.cpu(), vector_path)
            vectors_dict[layer_idx] = vector_path
        
        # Evaluate
        try:
            layer_results = run_injection_evaluation(
                model_wrapper, tokenizer, test_samples, vector,
                layer_idx, args.dataset, 1.0, args.max_new_tokens, 
                args.num_beams, args.use_early_stopping
            )
            
            diff = layer_results['accuracy'] - baseline_accuracy if baseline_accuracy else 0
            results.append({
                'layer': layer_idx,
                'accuracy': layer_results['accuracy'],
                'diff': diff,
                'correct': layer_results['correct'],
                'total': layer_results['total'],
            })
            
            print(f"  Layer {layer_idx:2d}: {layer_results['accuracy']:.2f}% "
                  f"({layer_results['correct']}/{layer_results['total']}) "
                  f"[{diff:+.2f}% vs baseline]")
            
        except Exception as e:
            print(f"  Layer {layer_idx:2d}: Evaluation Error - {e}")
            results.append({
                'layer': layer_idx,
                'accuracy': 0,
                'diff': -baseline_accuracy if baseline_accuracy else 0,
                'error': str(e),
            })
        
        # Clear CUDA cache after each layer
        torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"Method:   {args.method}")
    print(f"Dataset:  {args.dataset}")
    if baseline_accuracy:
        print(f"Baseline: {baseline_accuracy:.2f}%")
    print("-" * 70)
    
    # Filter out errors
    valid_results = [r for r in results if 'error' not in r]
    
    if valid_results:
        # Sort by accuracy
        valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Layer-wise average
        avg_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        print(f"\nLayer-wise Average: {avg_accuracy:.2f}%")
        
        print("\nTop 5 layers:")
        for r in valid_results[:5]:
            diff_str = f"({r['diff']:+.2f}%)" if baseline_accuracy else ""
            print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str}")
        
        if len(valid_results) > 5:
            print("\nBottom 5 layers:")
            for r in valid_results[-5:]:
                diff_str = f"({r['diff']:+.2f}%)" if baseline_accuracy else ""
                print(f"  Layer {r['layer']:2d}: {r['accuracy']:.2f}% {diff_str}")
        
        # Best layer
        best = valid_results[0]
        print(f"\n★ Best Layer: {best['layer']} with {best['accuracy']:.2f}%")
        if baseline_accuracy:
            print(f"  Improvement over baseline: {best['diff']:+.2f}%")
    
    # Error summary
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nErrors encountered in {len(error_results)} layers:")
        for r in error_results:
            print(f"  Layer {r['layer']}: {r['error'][:50]}...")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir, 
        f"layer_sweep_{args.method}_{args.dataset}_{timestamp}.txt"
    )
    
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Baseline: {baseline_accuracy:.2f}%\n" if baseline_accuracy else "Baseline: N/A\n")
        f.write(f"Support samples: {args.num_support_samples}\n")
        f.write(f"Test samples: {args.num_test_samples}\n")
        
        if args.method == "learnable":
            f.write(f"\nLearnable Config:\n")
            f.write(f"  Epochs: {args.num_epochs}\n")
            f.write(f"  Batch size: {args.batch_size}\n")
            f.write(f"  Grad accum: {args.gradient_accumulation_steps}\n")
            f.write(f"  Learning rate: {args.learning_rate}\n")
            f.write(f"  Lambda: {args.lambda_val}\n")
            f.write(f"  Max length: {args.max_length}\n")
        
        if args.method == "self_evolved":
            f.write(f"\nGRPO Config:\n")
            f.write(f"  Group size: {args.group_size}\n")
            f.write(f"  Iterations: {args.num_iterations}\n")
            f.write(f"  Questions/iter: {args.questions_per_iter}\n")
            f.write(f"  Learning rate: {args.grpo_lr}\n")
            f.write(f"  Temperature: {args.temperature}\n")
        
        f.write(f"\nLayer\tAccuracy\tDiff\tCorrect\tTotal\n")
        f.write("-" * 50 + "\n")
        
        for r in sorted(results, key=lambda x: x['layer']):
            if 'error' in r:
                f.write(f"{r['layer']}\tERROR\t-\t-\t-\t{r['error'][:30]}\n")
            else:
                f.write(f"{r['layer']}\t{r['accuracy']:.2f}\t{r['diff']:+.2f}\t"
                       f"{r['correct']}\t{r['total']}\n")
        
        # Summary at the end
        if valid_results:
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"Best Layer: {valid_results[0]['layer']} ({valid_results[0]['accuracy']:.2f}%)\n")
            f.write(f"Layer-wise Average: {avg_accuracy:.2f}%\n")
    
    print(f"\nResults saved to {result_file}")
    
    # Save vectors paths if saved
    if args.save_vectors and vectors_dict:
        vectors_file = os.path.join(
            args.output_dir,
            f"vectors_paths_{args.method}_{args.dataset}_{timestamp}.txt"
        )
        with open(vectors_file, "w") as f:
            for layer, path in sorted(vectors_dict.items()):
                f.write(f"Layer {layer}: {path}\n")
        print(f"Vector paths saved to {vectors_file}")
    
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
