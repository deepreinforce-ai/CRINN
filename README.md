
<div align="center">
  <picture>
      <img src="assets/header.png" width="60%" alt="CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning">
  </picture>
</div>

<hr>
<p align="center">
<a href="https://github.com/deepreinforce-ai/CRINN/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-GPL%20v3-blue.svg"/></a> &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>📄&nbsp;&nbsp;<a href="https://arxiv.org/abs/2507.14111">Paper</a></b>
</p>

# CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning



## Introduction

In this paper, we introduce **CRINN**, a novel reinforcement learning-augmented LLM framework for automated optimization of approximate nearest-neighbor search (ANNS) algorithms. CRINN employs contrastive reinforcement learning to automatically generate progressively faster ANNS implementations while maintaining accuracy constraints, transforming the traditionally manual and expertise-intensive optimization process into an automated search through the space of possible implementations.

CRINN achieves **best-in-class performance** on three out of six widely-used NNS benchmark datasets and matches state-of-the-art results on two others, demonstrating the effectiveness of RL-augmented LLMs for automating complex algorithmic optimizations.

<div align="center">
  <picture>
      <img src="assets/combined_datasets_plot.jpg" width="90%" alt="Evaluation Results">
  </picture>
<br>
 <p align="center">
    <strong>Fig</strong>：QPS versus recall curves for different models across six datasets. CINN achieves achieves best-in-class performance on three out of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular) and matching state-of-the-art results on two (SIFT-128 and GloVe-25)
</p>
</div>

### Key Features

- **Contrastive RL Framework**: Utilizes comparative analysis of code variants with execution metrics to learn effective optimization strategies
- **Automated ANNS Optimization**: Transforms manual optimization into automated search through implementation space
- **Superior Performance**: Achieves best performance on 3/6 benchmark datasets with improvements up to 85.25%
- **Modular Optimization**: Sequential optimization of three key modules: graph construction, search, and refinement
- **Broad Applicability**: Framework can be applied to any existing open-source ANNS algorithm as starting point



## How CRINN Works

### Core Architecture

CRINN operates through a **contrastive reinforcement learning** approach that:

1. **Analyzes Code Variants**: Performs comparative analysis of previously generated ANNS implementations alongside their execution metrics
2. **Learns Optimization Patterns**: Develops understanding of which code patterns lead to performance improvements vs. degradation
3. **Iterative Improvement**: Uses execution time as reward signal to drive LLM toward generating progressively more efficient implementations
4. **Modular Optimization**: Sequentially optimizes three key ANNS modules: graph construction, search, and refinement

### Optimization Strategy

**Starting Point**: Uses existing open-source ANNS algorithms (demonstrated with GLASS) as baseline **Sequential Modules**:

- **Graph Construction**: Adaptive search scaling, multi-level prefetching, multi-entry point architecture
- **Search**: Multi-tier entry point selection, batch processing with adaptive prefetching, intelligent early termination
- **Refinement**: Adaptive memory prefetching, pre-computed edge metadata with pattern recognition



## Technical Details

### Contrastive RL Training

- **Reward Function**: Area under QPS-recall curve in [0.85, 0.95] recall range
- **Training Method**: Group Relative Policy Optimization (GRPO)
- **Exemplar Selection**: Temperature-scaled softmax distribution based on performance scores
- **Training Data**: Exclusively trained on SIFT-128 dataset, evaluated across all datasets

### Key Optimization Discoveries

- **Adaptive Search Scaling**: Dynamic ef parameter adjustment based on recall requirements
- **Multi-Level Prefetching**: Intelligent prefetching considering neighbor density and search layer
- **Multi-Entry Point Architecture**: Parallel exploration from diverse entry points
- **Convergence Detection**: Smart early termination to avoid unnecessary exploration



## Usage

### Installation from Source

``` bash
sudo apt-get update && sudo apt-get install -y build-essential git python3 python3-distutils python3-venv
pip3 install numpy
pip3 install pybind11
bash build.sh
```



### Quick Start

```bash
python examples/main.py
```



## Limitations and Future Work

During the RL training process, we identified potential reward hacking scenarios where the model might exploit timing measurements or cache results. We've implemented safeguards against known cases, but welcome community feedback on additional edge cases.

**Future Directions**:

- Extend framework to other ANNS algorithms beyond GLASS baseline
- Explore multi-dataset training for improved generalization
- Integration with hardware-specific optimizations



## Citation

```latex
@article{li2025crinn,
  title={CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search},
  author={Li, Xiaoya and Sun, Xiaofei and Wang, Albert and Shum, Chris and Li, Jiwei},
  journal={arXiv preprint},
  year={2025}
}
```



## Contact

If you have any questions, please reach out to us at **research@deep-reinforce.com**.
