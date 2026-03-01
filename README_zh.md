<div align="center">

# 🤖 Aligned Agents, Biased Swarm：多智能体系统中的偏见放大测量

[![Paper](https://img.shields.io/badge/Paper-ArXiv-red?style=for-the-badge)](https://github.com/weizhihao1/MAS-Bias)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://github.com/weizhihao1/MAS-Bias)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/weizhihao1/MAS-Bias)

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

</div>

## 🔥 摘要

大语言模型（LLM）系统正在从单智能体流水线转向协作式多智能体系统（MAS）。尽管单个模型越来越“对齐”，本仓库关注一个关键的系统级问题：**协作会降低偏见，还是放大偏见？**

我们提出了一个开放式基准设定（基于人口属性的三选一对比选择），并评估多种 MAS 架构。核心发现是一致的：**即便单个智能体在隔离条件下相对中性，偏见仍会随着跨智能体推理传播而放大**。这一现象在角色分工、通信拓扑和更深层迭代系统中均可观察到。

<p align="center">
  <img src="assets/teaser0_1.png" alt="MAS 偏见放大示意图" width="900">
</p>



## 🧪 方法概览

### 1) 基准构建

不同于二元 yes/no 判断，每个问题提供 **三个具有不同人口属性的候选人（A/B/C）**。智能体需要输出对三者的概率分布与文本理由。

- 数据文件：
  - `data/implicit_prompts.json`
  - `data/explicit_prompts.json`

### 2) 指标

对于每个智能体输出分布，我们计算：

- **Gini 系数**（主要极化指标）
- 方差
- 熵
- 相对于均匀分布的 KL 散度

### 3) 评估架构

- 序列链式基线（`linear_*`）
- 拓扑变体（`parallel`、`spindle`）
- 深度/迭代设置（`iteration`，重复全连接单元）

## 📊 实验图示

### 基准分布

<p align="center">
  <img src="assets/bench_distribution.png" alt="基准分布" width="860">
</p>

### 主结果图（Part 1）

<p align="center">
  <img src="assets/1.png" alt="主结果图 1" width="860">
</p>

### 主结果图（Part 2）

<p align="center">
  <img src="assets/2.png" alt="主结果图 2" width="860">
</p>

### 触发脆弱性 / 扰动案例

<p align="center">
  <img src="assets/final.png" alt="触发脆弱性" width="860">
</p>

## 🚀 快速开始

### 1) 创建 conda 环境

创建 conda 环境并安装 Python 依赖：

```bash
conda create -n mas-bias python=3.11
conda activate mas-bias
pip install -r requirements.txt
```

### 2) 配置 API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

或使用自定义环境变量，并通过 CLI 传入：

```bash
python run_experiment.py --api-key-env YOUR_KEY_ENV
```

### 3) 使用默认配置运行

```bash
python run_experiment.py --config configs/default.json
```

### 4) 常用运行示例

```bash
# Persona 链式架构 + implicit 数据集
python run_experiment.py \
  --architecture linear_persona \
  --dataset-type implicit \
  --model-name gpt-4o-mini

# Iterative 架构，4 个迭代单元
python run_experiment.py \
  --architecture iteration \
  --num-iterations 4 \
  --dataset-type implicit

# Dry run（不调用 API），用于流程连通性检查
python run_experiment.py \
  --dry-run \
  --max-questions 5 \
  --save-interval 1
```

### 5) 实用 CLI 选项

```bash
python run_experiment.py --help
```

关键参数包括：

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

## 📁 项目结构

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

## 📦 输出文件

每次运行会在 `outputs/` 下生成带时间戳的目录，包含：

- `run_config.json`
- `*_question_metrics_progress_*.csv`
- `*_avg_metrics_progress_*.csv`
- `*_responses_progress_*.csv`
- `*_responses_progress_*.json`


## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weizhihao1/MAS-Bias&type=Date)](https://star-history.com/#weizhihao1/MAS-Bias&Date)

## Citation

如果你觉得这个项目有帮助，请引用论文：

```bibtex
@article{li2026agencybench,
  title={AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts},
  author={Li, Keyu and Shi, Junhao and Xiao, Yang and Jiang, Mohan and Sun, Jie and Wu, Yunze and Xia, Shijie and Cai, Xiaojie and Xu, Tianze and Si, Weiye and others},
  journal={arXiv preprint arXiv:2601.11044},
  year={2026}
}
```
