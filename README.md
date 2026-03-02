<div align="center">

# 🤖 Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems

<p><em>Official code, paper, and released artifacts for ICLR 2026.</em></p>

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](./_ICLR26__Bias.pdf)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/weizhihao1/MAS-Bias)

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

<p>
  <a href="#-recent-news">Recent News</a> •
  <a href="#-abstract">Abstract</a> •
  <a href="#-project-overview">Overview</a> •
  <a href="#-run-experiments">Run</a> •
  <a href="#-paper-setting--script-mapping">Mapping</a> •
  <a href="#-star-history">Star History</a>
</p>

</div>

## 🔥 Recent News

- **February 2026**: We are excited to announce that our paper has been accepted to **ICLR 2026**.
- **March 2026**: We are excited to announce the full open-source release, including:
  - GitHub repository (this codebase)
  - Hugging Face dataset: https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open

## 📖 Abstract

Large language model (LLM) systems are moving from single-agent pipelines to collaborative multi-agent systems (MAS). While individual models are increasingly aligned, this repository studies an important system-level question: **does collaboration reduce bias, or amplify it?**

We introduce an open-ended benchmark setting (three-way comparative choices across demographic attributes) and evaluate multiple MAS architectures. The key finding is consistent: **bias tends to amplify as reasoning propagates through agents**, even when individual agents look relatively neutral in isolation. This effect appears across role specialization, communication topology, and deeper iterative systems.

<p align="center">
  <img src="assets/teaser0_1.png" alt="Teaser: Bias amplification in MAS" width="900">
</p>

## 📦 Project Overview

| Component | Description |
| --- | --- |
| Paper | `_ICLR26__Bias.pdf` |
| Code | `mas-bias/` (all experiment scripts) |
| Dataset | `data/` (`implicit_prompts.json`, `explicit_prompts.json`) |
| Configs | `configs/*.env` |
| Launcher | `run_experiment.sh` |
| Results | `results/` |

## 📁 Repository Structure

```text
MAS-Bias/
├── _ICLR26__Bias.pdf
├── assets/
│   └── teaser0_1.png
├── configs/
│   ├── linear_plain.env
│   ├── linear_persona.env
│   ├── linear_function.env
│   ├── linear_mix.env
│   ├── spindle.env
│   ├── parallel.env
│   ├── ffn.env
│   ├── iteration.env
│   └── different_model.env
├── data/
│   ├── implicit_prompts.json
│   └── explicit_prompts.json
├── mas-bias/
│   ├── runtime_config.py
│   ├── linear_plain.py
│   ├── linear_persona.py
│   ├── linear_function.py
│   ├── linear_mix.py
│   ├── spindle.py
│   ├── parallel.py
│   ├── ffn.py
│   ├── iteration.py
│   └── different_model.py
├── results/
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── run_experiment.sh
├── README.md
└── README_zh.md
```

## 🧪 Environment Setup

### Option A: UV (project name: `mas-bias`)

```bash
uv sync
source .venv/bin/activate
```

### Option B: Conda (env name: `mas-bias`)

```bash
conda env create -f environment.yml
conda activate mas-bias
```

## ⚙️ Configuration and Credentials

Edit `configs/*.env` directly for each setup:

- `MODEL_NAME`: model identifier (single-model settings)
- `API_KEY_ENV`: env var name holding your API key
- `BASE_URL`: optional API base URL
- `DATASET_TYPE`: `implicit` or `explicit`
- `MIXED_AGENT_MODELS_JSON`: per-agent model mapping (`different_model` only)

Set API key before running:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🚀 Run Experiments

List all configurations:

```bash
./run_experiment.sh --list
```

Run one setting:

```bash
./run_experiment.sh linear_plain
```

Override parameters without editing files:

```bash
./run_experiment.sh linear_persona DATASET_TYPE=explicit MODEL_NAME=gpt-4o-mini
```

## 🧭 Paper Setting ↔ Script Mapping

| Paper Item | Config | Script |
| --- | --- | --- |
| Figure 5(a) | `linear_plain` | `mas-bias/linear_plain.py` |
| Figure 5(b) | `linear_persona` | `mas-bias/linear_persona.py` |
| Figure 5(c) | `linear_function` | `mas-bias/linear_function.py` |
| Figure 5(d) | `linear_mix` | `mas-bias/linear_mix.py` |
| Figure 6(a) | `spindle` | `mas-bias/spindle.py` |
| Figure 6(b) | `parallel` | `mas-bias/parallel.py` |
| Figure 6(c) | `ffn` | `mas-bias/ffn.py` |
| Figure 6(d) | `iteration` | `mas-bias/iteration.py` |
| Table 1 | `different_model` | `mas-bias/different_model.py` |

## 📊 Output Locations

Each run writes progress CSV files to one result folder (at repo root):

- `linear_plain_results/`
- `linear_persona_results/`
- `linear_function_results/`
- `linear_mix_results/`
- `spindle_results/`
- `parallel_results/`
- `ffn_results/`
- `iteration_results/`
- `different_results/`

## 🤗 Dataset

- Hugging Face: https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open
- Local JSON: `data/implicit_prompts.json`, `data/explicit_prompts.json`

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weizhihao1/MAS-Bias&type=Date)](https://star-history.com/#weizhihao1/MAS-Bias&Date)

## Citation

If you find this project useful, please cite:

```bibtex
@article{li2026agencybench,
  title={AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts},
  author={Li, Keyu and Shi, Junhao and Xiao, Yang and Jiang, Mohan and Sun, Jie and Wu, Yunze and Xia, Shijie and Cai, Xiaojie and Xu, Tianze and Si, Weiye and others},
  journal={arXiv preprint arXiv:2601.11044},
  year={2026}
}
```
