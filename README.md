# CoT Vectors Reproduction

Reproduction of "CoT Vectors: Transferring and Probing the Reasoning Mechanisms of LLMs"

## Quick Start

```bash
# Extracted CoT Vector
python main.py \
    --model_path /home/haichao/TA/cotv/models/Qwen2.5-Math-7B \
    --data_path /home/haichao/TA/cotv/data \
    --dataset gsm8k \
    --method extracted \
    --layer_idx 0 \
    --mode both

# Learnable CoT Vector
python main.py \
    --model_path /home/haichao/TA/cotv/models/Qwen2.5-Math-7B \
    --data_path /home/haichao/TA/cotv/data \
    --dataset gsm8k \
    --method learnable \
    --layer_idx 0 \
    --mode both
```

## Speed Options

| Config | Command | Speed | Accuracy |
|--------|---------|-------|----------|
| Paper (default) | `--num_beams 3` | ~42s/sample | Best |
| Fast | `--num_beams 1` | ~23s/sample | Slightly lower |
| Balanced | `--num_beams 3 --max_new_tokens 450` | ~38s/sample | Same |

## Layer Sweep

```bash
python run_layer_sweep.py \
    --model_path /home/haichao/TA/cotv/models/Qwen2.5-Math-7B \
    --data_path /home/haichao/TA/cotv/data \
    --dataset gsm8k \
    --num_support_samples 100 \
    --num_test_samples 200
```

## Project Structure

```
cot_vectors/
├── main.py              # Entry point
├── run_layer_sweep.py   # Layer sweep script
├── config/
│   └── secrets.yaml     # WandB credentials
├── src/
│   ├── args.py          # Arguments
│   ├── data_utils.py    # Data loading
│   ├── models.py        # Model wrapper
│   ├── eval.py          # Evaluation
│   ├── utils.py         # Utilities
│   └── methods/
│       ├── base.py
│       ├── extracted.py # Extracted CoT Vector
│       └── learnable.py # Learnable CoT Vector
└── requirements.txt
```

## Key Parameters

- `--layer_idx`: Layer for vector injection (0 = first layer, recommended for learnable)
- `--scaling_factor`: Scaling factor μ (default 1.0)
- `--lambda_val`: Loss balance factor λ (default 0.5, for learnable method)
- `--num_beams`: Beam search width (3 = paper setting, 1 = fast)
