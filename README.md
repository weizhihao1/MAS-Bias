<div align="center">

# 🤖 Aligned Agents, Biased Swarm: Measuring Bias Amplification in Multi-Agent Systems

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red?style=for-the-badge)](https://github.com/weizhihao1/MAS-Bias)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://github.com/weizhihao1/MAS-Bias)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/weizhihao1/MAS-Bias)

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

</div>

## 🔥 Abstract

Large language model (LLM) systems are moving from single-agent pipelines to collaborative multi-agent systems (MAS). While individual models are increasingly aligned, this repository studies an important system-level question: **does collaboration reduce bias, or amplify it?**

We introduce an open-ended benchmark setting (three-way comparative choices across demographic attributes) and evaluate multiple MAS architectures. The key finding is consistent: **bias tends to amplify as reasoning propagates through agents**, even when individual agents look relatively neutral in isolation. This effect appears across role specialization, communication topology, and deeper iterative systems.

<p align="center">
  <img src="assets/teaser0_1.png" alt="Teaser: Bias amplification in MAS" width="900">
</p>



## 🧪 Method Overview 

### 1) Benchmark formulation

Instead of binary yes/no judgments, each question provides **three demographically distinct protagonists (A/B/C)**. Agents must output a probability distribution over choices and a textual rationale.

- Dataset files:
  - `data/implicit_prompts.json`
  - `data/explicit_prompts.json`

### 2) Metrics

For each agent output distribution, we compute:

- **Gini coefficient** (main polarization metric)
- Variance
- Entropy
- KL divergence to uniform distribution

### 3) Architectures evaluated

- Sequential chain baselines (`linear_*`)
- Topology variants (`parallel`, `spindle`)
- Depth/iteration setting (`iteration`, repeated fully-connected unit)

## 📊 Experimental Figures

### Benchmark distribution

<p align="center">
  <img src="assets/bench_distribution.png" alt="Benchmark Distribution" width="860">
</p>

### Main results figure (part 1)

<p align="center">
  <img src="assets/1.png" alt="Main Results 1" width="860">
</p>

### Main results figure (part 2)

<p align="center">
  <img src="assets/2.png" alt="Main Results 2" width="860">
</p>

### Trigger vulnerability / perturbation case

<p align="center">
  <img src="assets/final.png" alt="Trigger Vulnerability" width="860">
</p>

## 🚀 Quick Start

### 1) Create conda environment

Create a conda environment and install Python dependencies:

```bash
conda create -n mas-bias python=3.11
conda activate mas-bias
pip install -r requirements.txt
```

### 2) Configure API key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or set any custom environment variable and pass it by CLI:

```bash
python run_experiment.py --api-key-env YOUR_KEY_ENV
```

### 3) Run with default config

```bash
python run_experiment.py --config configs/default.json
```

### 4) Common run examples

```bash
# Persona chain on implicit set
python run_experiment.py \
  --architecture linear_persona \
  --dataset-type implicit \
  --model-name gpt-4o-mini

# Iterative architecture with 4 units
python run_experiment.py \
  --architecture iteration \
  --num-iterations 4 \
  --dataset-type implicit

# Dry run (no API call), useful for pipeline sanity checks
python run_experiment.py \
  --dry-run \
  --max-questions 5 \
  --save-interval 1
```

### 5) Useful CLI options

```bash
python run_experiment.py --help
```

Key options include:

- `--architecture`
- `--dataset-type`
- `--model-name`
- `--base-url`
- `--api-key-env`
- `--num-iterations`
- `--save-interval`
- `--max-questions`
- `--data-dir`
- `--output-dir`

## 📁 Project Structure

```text
MAS-Bias/
├── assets/
├── configs/
│   └── default.json
├── data/
│   ├── explicit_prompts.json
│   └── implicit_prompts.json
├── mas_bias/
│   ├── cli.py
│   ├── config.py
│   ├── constants.py
│   ├── metrics.py
│   ├── parsing.py
│   ├── prompts.py
│   └── runner.py
├── environment.yml
├── run_experiment.py
├── requirements.txt
└── README.md
```

## 📦 Output Files

Each run creates a timestamped folder under `outputs/`, containing:

- `run_config.json`
- `*_question_metrics_progress_*.csv`
- `*_avg_metrics_progress_*.csv`
- `*_responses_progress_*.csv`
- `*_responses_progress_*.json`


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weizhihao1/MAS-Bias&type=Date)](https://star-history.com/#weizhihao1/MAS-Bias&Date)

## Citation

If you find this project useful, please cite the paper:

```bibtex
@article{li2026agencybench,
  title={AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts},
  author={Li, Keyu and Shi, Junhao and Xiao, Yang and Jiang, Mohan and Sun, Jie and Wu, Yunze and Xia, Shijie and Cai, Xiaojie and Xu, Tianze and Si, Weiye and others},
  journal={arXiv preprint arXiv:2601.11044},
  year={2026}
}
```
