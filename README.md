# Tool-Reflection-Bench

**Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions**

[![arXiv](https://img.shields.io/badge/arXiv-2509.18847-b31b1b.svg)](https://arxiv.org/abs/2509.18847)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Tool-Reflection-Bench is a benchmark and training framework for improving LLM tool-calling accuracy through **structured reflection**. When a tool call fails, the model learns to diagnose the error in a `<reflect>` block and issue a corrected call, rather than blindly retrying.

The training uses the combination of **GSPO** and **DAPO**  with a custom reward function that decomposes correctness into three components:

- **Reflection quality** (`s_ref`): How well the model diagnoses the error
- **Call correctness** (`s_call`): Whether the corrected tool call matches the ground truth
- **Final answer quality** (`s_final`): End-to-end task completion

This is combined with a format factor that penalizes missing/malformed tags and call count mismatches.

## Key Results

Repair@N measures the percentage of failed tool calls that the model successfully corrects.

| Model | Method | Repair@1 | Repair@3 | Repair@5 |
|-------|--------|----------|----------|----------|
| Qwen2.5-7B | Base | 2.4% | 6.1% | 8.0% |
| Qwen2.5-7B | **Ours** | **9.3%** | **10.3%** | **11.4%** |
| Llama-3.1-8B | Base | 0.7% | 5.1% | 6.8% |
| Llama-3.1-8B | **Ours** | **4.7%** | **20.5%** | **26.4%** |
| Qwen3-4B | Base | 9.6% | 10.6% | 10.6% |
| Qwen3-4B | **Ours** | **14.9%** | **18.5%** | **19.5%** |

## Installation

```bash
# Clone the repository
git clone https://github.com/MeiGen-AI/Tool-Reflection-Bench.git
cd Tool-Reflection-Bench

# Create a virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**:
> - The bundled `train/swift` subtree is imported via `PYTHONPATH` in the provided training and evaluation scripts, so no separate editable install is required.
> - vLLM requires CUDA-compatible GPUs. For evaluation only, a single GPU with 24GB+ VRAM is sufficient for 7B models. Training requires 8x 80GB GPUs.

## Project Structure

```
Tool-Reflection-Bench/
├── README.md
├── requirements.txt
├── .gitignore
├── train/                          # Training code
│   ├── train.sh                    # One-click training launcher
│   ├── grpo_training_script.py     # Main GRPO training script
│   ├── merge_lora.py               # LoRA adapter merging
│   ├── draw.py                     # Reward curve visualization
│   └── swift/                      # Bundled modified copy of ms-swift (Apache-2.0)
│       └── plugin/
│           ├── orm.py              # Reward functions (core)
│           └── toolrl_reward.py    # ToolRL reward baseline
├── data/                           # Training data
│   ├── train_qwen.jsonl            # Qwen-format training data (4928 samples)
│   ├── train_llama.jsonl           # Llama-format training data (4928 samples)
└── benchmark/                      # Evaluation
    ├── eval.sh                     # One-click evaluation launcher
    ├── test.py                     # Evaluation script
    ├── test_qwen_1000.jsonl        # Qwen test set (1000 samples)
    └── test_llama_1000.jsonl       # Llama test set (1000 samples)
```

## Training

### Quick Start

```bash
cd train

# Train Qwen2.5-7B (auto-selects Qwen training data)
bash train.sh --model /path/to/Qwen2.5-7B-Instruct

# Train Llama-3.1-8B (auto-selects Llama training data)
bash train.sh --model /path/to/Llama-3.1-8B-Instruct

# Train Qwen3-4B with custom GPU count
bash train.sh --model /path/to/Qwen3-4B-Instruct --gpus 4
```

### Supported Models

| Shorthand | HuggingFace ID | Type |
|-----------|---------------|------|
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | Qwen |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct` | Qwen3 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Llama |

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA rank | default | Parameter-efficient fine-tuning |
| Learning rate | 1e-5 | With 5% warmup |
| Epochs | 1 | Single pass |
| Temperature | 0.8 | Generation sampling |
| `num_generations` | 4 | GRPO group size |
| `epsilon` | 0.2 | PPO clip range |
| `epsilon_high` | 0.28 | DAPO upper clip |
| `beta` | 0.05 | KL penalty coefficient |
| `max_completion_length` | 1024 | Max generation tokens |
| `importance_sampling_level` | sequence | GSPO sequence-level |

### Merge LoRA Weights

After training, merge the adapter weights into a full model:

```bash
python train/merge_lora.py \
    --base_model /path/to/Qwen2.5-7B-Instruct \
    --adapter_ckpt outputs/Qwen2.5-7B-Instruct-GRPO/checkpoint-1000 \
    --save_dir outputs/Qwen2.5-7B-Instruct-GRPO/checkpoint-1000-merged
```

### Visualize Training Curves

```bash
python train/draw.py  # Edit the logging.jsonl path inside the script
```

## Evaluation

### Local Model (vLLM)

```bash
cd benchmark

# Pass@1 evaluation
bash eval.sh --model /path/to/model --n 1 --tp 4

# Pass@5 evaluation
bash eval.sh --model /path/to/model --n 5 --tp 8

# With custom batch size and GPU memory
bash eval.sh --model /path/to/model --n 3 --tp 4 --batch 32 --gpu-mem 0.85
```

### OpenAI-Compatible API

```bash
# Using OpenAI API
export OPENAI_API_KEY="your-key"
bash eval.sh --api gpt-4o --n 3

# Any OpenAI-compatible endpoint
bash eval.sh --api Qwen/Qwen2.5-72B-Instruct \
    --api-key "$API_KEY" \
    --base-url "https://your-openai-compatible-endpoint/v1"
```

### Direct Python Usage

```bash
python benchmark/test.py --model /path/to/model --n 5 --tp 4
python benchmark/test.py --api gpt-4o --n 1
```

### Metrics

- **Repair@N (Pass@N)**: The question is considered passed if **any** of the N attempts produces a fully correct tool call correction (exact match on function name and all arguments).
- **Average Score**: Mean per-attempt score across all questions, reflecting partial credit.

## Training Data

| Perturbation | Description |
|-------------|-------------|
| **Argument Error** | Corrupt argument values (wrong type, sentinel strings, null) |
| **Call Order Swap** | Swap the order of tool calls, causing dependency errors |
| **Missing Call** | Remove a prerequisite call, breaking downstream calls |
| **Redundant Call** | Duplicate a call, producing redundant results |

Each sample follows the format:
```
system → user → [correct calls...] → erroneous call → error response
```

The ground truth contains:
```
<reflect>Error diagnosis and correction plan</reflect>

<tool_call>
{"name": "function_name", "arguments": {"param": "correct_value"}}
</tool_call>
```

This open-source release includes the processed training and evaluation datasets only. The internal data-generation scripts are not part of this repository.

## Reward Design

The reward function follows a **Structured Factor** (S-F) decomposition:

```
S = (w_r * I_r * s_ref + w_c * I_c * s_call + w_f * I_f * s_final) / W_active
F = clip[0,1](1 - lambda_m * P_total * r_fmt)
R = S * F
```

Where:
- `s_ref` (weight 0.1): Semantic similarity of `<reflect>` block to ground truth
- `s_call` (weight 0.7): Binary — exact match of tool calls (function name + arguments)
- `s_final` (weight 0.2): Semantic similarity of final answer to ground truth
- `F`: Format factor penalizing missing/extra XML tags and call count mismatches

If the core reward `R < epsilon`, a backoff reward is computed as `0.15 * Sim(generated, ground_truth)`.

## Citation

```bibtex
@article{su2025failure,
  title={Failure makes the agent stronger: Enhancing accuracy through structured reflection for reliable tool interactions},
  author={Su, Junhao and Wan, Yuanliang and Yang, Junwei and Shi, Hengyu and Han, Tianyang and Luo, Junfeng and Qiu, Yurui},
  journal={arXiv preprint arXiv:2509.18847},
  year={2025}
}
```

## License

The original Tool-Reflection-Bench code in this repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

This repository also contains a bundled modified copy of `ms-swift` under `train/swift/`. That third-party subtree remains under the Apache License 2.0 with its original copyright notices retained in the source files. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) for details.
