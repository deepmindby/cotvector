import os
import re
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import numpy as np
from transformers import AutoModelForCausalLM

# 导入调整：尝试导入 load_tokenizer
try:
    from src.models import CoTModelWrapper, load_tokenizer
except ImportError:
    # Fallback 如果 src.models 里没有显式导出 load_tokenizer
    from src.models import CoTModelWrapper
    from transformers import AutoTokenizer
    def load_tokenizer(path):
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)

from src.data_utils import load_dataset
from src.utils import extract_answer_from_text, compare_answers, set_seed

def parse_filename(filename):
    """
    解析文件名，提取方法、数据集和层数。
    """
    name = os.path.splitext(filename)[0]
    
    # 匹配层数 (L数字)
    layer_match = re.search(r'_L(\d+)', name)
    if not layer_match:
        return None
    layer_idx = int(layer_match.group(1))
    
    if name.startswith("extracted"):
        method = "extracted"
    elif name.startswith("learnable"):
        method = "learnable"
    elif name.startswith("self_evolved"):
        method = "self_evolved"
    else:
        method = "unknown"
        
    return {
        "filename": filename,
        "method": method,
        "layer": layer_idx,
        "path": os.path.join("outputs", filename)
    }

def analyze_vector_stats(vector_path, device):
    """统计向量的物理属性：尺寸、范数、分布"""
    try:
        # 加载向量时指定 device，避免跨设备问题
        vec = torch.load(vector_path, map_location=device)
        
        # 兼容: 如果保存的是dict (包含metadata)，则提取vector字段
        if isinstance(vec, dict) and "vector" in vec:
            vec = vec["vector"]
        
        # 确保是 1D tensor
        if vec.dim() > 1:
            vec = vec.view(-1)
            
        stats = {
            "norm": vec.norm(p=2).item(),
            "mean": vec.mean().item(),
            "std": vec.std().item(),
            "max": vec.max().item(),
            "min": vec.min().item(),
            "dim": vec.shape[0]
        }
        return stats, vec
    except Exception as e:
        print(f"Error loading vector {vector_path}: {e}")
        return None, None

def evaluate_reasoning(model_wrapper, vector, layer_idx, samples, args):
    """
    核心评估逻辑：
    不仅看准确率，还要看推理长度 (Token Count)，以此判断是否走捷径。
    """
    results = {
        "correct": 0,
        "total": 0,
        "total_tokens": 0,
        "example_generations": [] 
    }
    
    # 注册向量注入 Hook
    # 注意：根据 models.py 定义，register_injection_hook 需要 vector, scaling_factor, requires_grad
    # 这里我们只评估，所以 scaling_factor=1.0, requires_grad=False
    hook = model_wrapper.register_injection_hook(layer_idx, vector, scaling_factor=1.0, requires_grad=False)
    
    try:
        for i, sample in enumerate(tqdm(samples, desc=f"Eval Layer {layer_idx}", leave=False)):
            question = sample['question'] if isinstance(sample, dict) else sample.question
            gold_answer = sample['answer'] if isinstance(sample, dict) else sample.answer
            
            # 构建 Prompt
            prompt = f"Question: {question}\nLet's think step by step."
            
            # 【修复】显式构建 attention_mask 并移动到正确设备
            inputs = model_wrapper.model.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model_wrapper.device)
            attention_mask = inputs.attention_mask.to(model_wrapper.device)

            # 生成
            try:
                with torch.no_grad():
                    output_ids = model_wrapper.model.generate(
                        input_ids,
                        attention_mask=attention_mask, # 【修复】传入 mask
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False, # Greedy decoding specifically for evaluation consistency
                        pad_token_id=model_wrapper.model.tokenizer.pad_token_id,
                        eos_token_id=model_wrapper.model.tokenizer.eos_token_id
                    )
                
                # 解码
                new_tokens = output_ids[0, input_ids.shape[1]:]
                output_text = model_wrapper.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
            except Exception as e:
                print(f"Generation error: {e}")
                output_text = ""

            # 统计生成长度
            gen_len = len(new_tokens) if 'new_tokens' in locals() else 0
            
            # 提取答案并对比
            pred_answer = extract_answer_from_text(output_text)
            is_correct = compare_answers(pred_answer, gold_answer)
            
            results["total"] += 1
            if is_correct:
                results["correct"] += 1
            results["total_tokens"] += gen_len
            
            # 保存少量的样本供人工检阅
            if i < 2: 
                results["example_generations"].append({
                    "q": question,
                    "gold": gold_answer,
                    "pred": pred_answer,
                    "full_text": output_text[-300:] 
                })
                
    finally:
        # 清理 Hook
        model_wrapper.clear_hooks()
        
    avg_len = results["total_tokens"] / results["total"] if results["total"] > 0 else 0
    accuracy = results["correct"] / results["total"] if results["total"] > 0 else 0
    
    return accuracy, avg_len, results["example_generations"]

def get_baseline_performance(model_wrapper, samples, args):
    """获取不注入向量时的基准表现"""
    print("Running Baseline Evaluation (No Vectors)...")
    results = {"correct": 0, "total": 0, "total_tokens": 0}
    
    model_wrapper.clear_hooks()
    
    for sample in tqdm(samples, desc="Baseline"):
        question = sample['question'] if isinstance(sample, dict) else sample.question
        gold_answer = sample['answer'] if isinstance(sample, dict) else sample.answer
        
        prompt = f"Question: {question}\nLet's think step by step."
        
        # 【修复】显式构建 attention_mask 并移动到正确设备
        inputs = model_wrapper.model.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model_wrapper.device)
        attention_mask = inputs.attention_mask.to(model_wrapper.device)
        
        with torch.no_grad():
            output_ids = model_wrapper.model.generate(
                input_ids,
                attention_mask=attention_mask, # 【修复】传入 mask
                max_new_tokens=args.max_new_tokens, 
                do_sample=False,
                pad_token_id=model_wrapper.model.tokenizer.pad_token_id,
                eos_token_id=model_wrapper.model.tokenizer.eos_token_id
            )
        
        new_tokens = output_ids[0, input_ids.shape[1]:]
        output_text = model_wrapper.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        gen_len = len(new_tokens)
        pred_answer = extract_answer_from_text(output_text)
        is_correct = compare_answers(pred_answer, gold_answer)
        
        results["total"] += 1
        if is_correct: results["correct"] += 1
        results["total_tokens"] += gen_len
        
    avg_len = results["total_tokens"] / results["total"]
    accuracy = results["correct"] / results["total"]
    return accuracy, avg_len

def main():
    parser = argparse.ArgumentParser(description="Deep Evaluation of CoT Vectors")
    parser.add_argument("--model_path", type=str, default="/home/haichao/TA/CoTVRL/models/Qwen2.5-Math-7B")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["qwen", "llama"], help="Model architecture name")
    parser.add_argument("--data_path", type=str, default="/home/haichao/TA/CoTVRL/data")
    parser.add_argument("--vector_dir", type=str, default="outputs")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--num_samples", type=int, default=50, help="评估样本数量")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--target_methods", nargs="+", default=["extracted", "learnable", "self_evolved"])
    parser.add_argument("--layers", nargs="+", type=int, help="指定层数")
    
    args = parser.parse_args()
    set_seed(42)
    
    # 1. 扫描向量文件
    if not os.path.exists(args.vector_dir):
        print(f"Error: Vector directory {args.vector_dir} not found.")
        return

    files = [f for f in os.listdir(args.vector_dir) if f.endswith(".pt")]
    vector_configs = []
    for f in files:
        info = parse_filename(f)
        if info and info['method'] in args.target_methods:
            if args.layers and info['layer'] not in args.layers:
                continue
            vector_configs.append(info)
            
    vector_configs.sort(key=lambda x: (x['method'], x['layer']))
    
    print(f"Found {len(vector_configs)} vectors to evaluate.")
    
    # 2. 加载模型和Tokenizer
    print(f"Loading Tokenizer from {args.model_path}...")
    tokenizer = load_tokenizer(args.model_path)
    
    print(f"Initializing Model Wrapper (loading model)...")
    model_wrapper = CoTModelWrapper(args.model_path, args.model_name)
    
    # 手动把 tokenizer 绑到 model 上，方便后续 evaluate_reasoning 调用
    model_wrapper.model.tokenizer = tokenizer

    # 【重要修复】删除手动移动到 CUDA 的代码
    # 因为 device_map="auto" 已经由 accelerate 托管了设备
    # if torch.cuda.is_available():
    #     print("Moving model to CUDA...")
    #     model_wrapper.model.to("cuda")  <-- 已删除
    
    # 打印设备分布信息，确认加载正常
    print(f"Model device map: {model_wrapper.model.hf_device_map if hasattr(model_wrapper.model, 'hf_device_map') else 'Single Device'}")
    
    # 获取主设备用于存放向量 (通常是 device_map 的第一个设备)
    device = model_wrapper.device 

    print(f"Loading dataset {args.dataset}...")
    dataset = load_dataset(args.data_path, args.dataset, split="test", num_samples=args.num_samples)
    
    # 3. 运行 Baseline
    base_acc, base_len = get_baseline_performance(model_wrapper, dataset, args)
    print(f"\nBaseline | Acc: {base_acc:.2%} | Avg Len: {base_len:.1f} tokens")
    
    # 4. 评估每个向量
    report_data = []
    
    for config in vector_configs:
        print(f"\nEvaluating {config['filename']}...")
        
        # 统计向量
        stats, vec = analyze_vector_stats(config['path'], device)
        if vec is None: continue

        # 评估
        acc, avg_len, examples = evaluate_reasoning(model_wrapper, vec, config['layer'], dataset, args)
        
        len_change_ratio = (avg_len - base_len) / base_len if base_len > 0 else 0
        acc_change = acc - base_acc
        
        diagnosis = "Neutral"
        if acc_change > 0.05:
            if len_change_ratio < -0.3:
                diagnosis = "⚠️ Shortcut (Acc Up, Len Down)"
            elif len_change_ratio > -0.05:
                diagnosis = "✅ Real CoT (Acc Up, Len Stable/Up)"
        elif acc_change < -0.1:
            diagnosis = "❌ Damaged"
            
        row = {
            "Method": config['method'],
            "Layer": config['layer'],
            "Norm": f"{stats['norm']:.2f}",
            "Acc": f"{acc:.2%}",
            "Acc Delta": f"{acc_change:+.2%}",
            "Len": f"{avg_len:.0f}",
            "Len Delta": f"{len_change_ratio:+.1%}",
            "Diagnosis": diagnosis
        }
        report_data.append(row)
        
        if abs(acc_change) > 0.05:
            print(f"--- Sample Output ({config['method']} L{config['layer']}) ---")
            if examples:
                print(f"Q: {examples[0]['q'][:100]}...")
                print(f"Pred: ...{examples[0]['full_text']}")
            print("---------------------------------------------")

    print("\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    print(f"Baseline Accuracy: {base_acc:.2%} | Baseline Avg Length: {base_len:.0f}")
    
    df = pd.DataFrame(report_data)
    if not df.empty:
        print(tabulate(df, headers="keys", tablefmt="grid"))
        df.to_csv(os.path.join(args.vector_dir, "deep_eval_report.csv"), index=False)
        print(f"\nReport saved to {os.path.join(args.vector_dir, 'deep_eval_report.csv')}")
    else:
        print("No results to report.")

if __name__ == "__main__":
    main()
