<div align="center">

# 🤖 Aligned Agents, Biased Swarm：多智能体系统中的偏见放大测量

<p><em>ICLR 2026 论文配套代码、数据与开放实验产物。</em></p>

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=for-the-badge)](./_ICLR26__Bias.pdf)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open)
[![Code](https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/weizhihao1/MAS-Bias)

<p align="center">
  <a href="./README.md">English</a> | <a href="./README_zh.md">简体中文</a>
</p>

<p>
  <a href="#-recent-news">Recent News</a> •
  <a href="#-摘要">摘要</a> •
  <a href="#-项目概览">概览</a> •
  <a href="#-运行实验">运行</a> •
  <a href="#-论文设定--脚本映射">映射</a> •
  <a href="#-star-history">Star History</a>
</p>

</div>

## 🔥 Recent News

- **2026 年 2 月**：我们非常激动地宣布，论文已被 **ICLR 2026** 正式接收。
- **2026 年 3 月**：我们非常激动地宣布项目已彻底开源，包括：
  - GitHub 代码仓库（本仓库）
  - Hugging Face 数据集：https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open

## 📖 摘要

大语言模型（LLM）系统正在从单智能体流水线转向协作式多智能体系统（MAS）。尽管单个模型越来越“对齐”，本仓库关注一个关键的系统级问题：**协作会降低偏见，还是放大偏见？**

我们提出了一个开放式基准设定（基于人口属性的三选一对比选择），并评估多种 MAS 架构。核心发现是一致的：**即便单个智能体在隔离条件下相对中性，偏见仍会随着跨智能体推理传播而放大**。这一现象在角色分工、通信拓扑和更深层迭代系统中均可观察到。

<p align="center">
  <img src="assets/teaser0_1.png" alt="MAS 偏见放大示意图" width="900">
</p>

## 📦 项目概览

| 组件 | 说明 |
| --- | --- |
| 论文 | `_ICLR26__Bias.pdf` |
| 代码 | `mas-bias/`（全部实验脚本） |
| 数据 | `data/`（`implicit_prompts.json`、`explicit_prompts.json`） |
| 配置 | `configs/*.env` |
| 启动脚本 | `run_experiment.sh` |
| 结果文件 | `results/` |

## 📁 仓库结构

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

## 🧪 环境构建

### 方式 A：UV（项目名：`mas-bias`）

```bash
uv sync
source .venv/bin/activate
```

### 方式 B：Conda（环境名：`mas-bias`）

```bash
conda env create -f environment.yml
conda activate mas-bias
```

## ⚙️ 配置与密钥

每个实验可直接修改 `configs/*.env`：

- `MODEL_NAME`：模型名称（单模型设定）
- `API_KEY_ENV`：API Key 对应的环境变量名
- `BASE_URL`：可选 API Base URL
- `DATASET_TYPE`：`implicit` 或 `explicit`
- `MIXED_AGENT_MODELS_JSON`：按 agent 指定模型（仅 `different_model`）

运行前请先设置 API Key：

```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🚀 运行实验

查看全部可运行配置：

```bash
./run_experiment.sh --list
```

运行某个设定：

```bash
./run_experiment.sh linear_plain
```

临时覆盖参数（无需修改配置文件）：

```bash
./run_experiment.sh linear_persona DATASET_TYPE=explicit MODEL_NAME=gpt-4o-mini
```

## 🧭 论文设定 ↔ 脚本映射

| 论文结果 | 配置名 | 脚本 |
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

## 📊 输出结果位置

每次运行会在仓库根目录生成对应结果目录（CSV 进度文件）：

- `linear_plain_results/`
- `linear_persona_results/`
- `linear_function_results/`
- `linear_mix_results/`
- `spindle_results/`
- `parallel_results/`
- `ffn_results/`
- `iteration_results/`
- `different_results/`

## 🤗 数据集

- Hugging Face：https://huggingface.co/datasets/weizhihao1/Discrim-Eval-Open
- 本地 JSON：`data/implicit_prompts.json`、`data/explicit_prompts.json`

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=weizhihao1/MAS-Bias&type=Date)](https://star-history.com/#weizhihao1/MAS-Bias&Date)

## Citation

如果你觉得这个项目有帮助，请引用：

```bibtex
@article{li2026agencybench,
  title={AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts},
  author={Li, Keyu and Shi, Junhao and Xiao, Yang and Jiang, Mohan and Sun, Jie and Wu, Yunze and Xia, Shijie and Cai, Xiaojie and Xu, Tianze and Si, Weiye and others},
  journal={arXiv preprint arXiv:2601.11044},
  year={2026}
}
```
