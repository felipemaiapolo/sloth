# Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families

Welcome to our GitHub repository! This repository is based on the ideas introduced in

[Maia Polo, Felipe, Seamus Somerstep, Leshem Choshen, Yuekai Sun, and Mikhail Yurochkin. "Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families." arXiv preprint arXiv:2412.06540 (2024).](https://arxiv.org/abs/2412.06540)

## Overview

Traditional scaling laws for LLMs rely on parameters like model size and training tokens to estimate performance. However, due to differences in training configurations and data processing, these laws often fail to generalize across diverse model families. Family-specific scaling laws provide more accuracy but require costly training of multiple models with varying sizes for each family. Slothproposes a skill-based framework that assumes LLM performance is driven by **low-dimensional latent skills** (e.g., reasoning, instruction following). These latent skills are produced from computational resources (e.g., model size, training tokens) with different efficiencies depending on the model family.

By leveraging publicly available benchmark data and the inherent correlations across benchmarks, Sloth provides:  
- **Accurate predictions** of LLM performance without extensive model training.  
- **Interpretability**, offering insights into scaling behaviors for downstream tasks.  

## Installation

To use the code in this repository, clone the repo and create a conda environment using:

```
conda env create --file=sloth.yaml
conda activate sloth
```

##  Quick start

If you are interested in checking how Sloth is fitted and interpreted, please check [this notebook](https://github.com/felipemaiapolo/sloth/notebooks/interpretability_plots.ipynb).


## Reproducing results from the paper

1. For the prediction experiments, run `python experiments.py --min_models j` for `j=2` and `j=3`. Then, use `results_plots.ipynb` to generate the performance heatmaps. 
2. To generate the rest of the plots in the paper, please run the notebooks with ending `_plots.ipynb`.


## Citing

```
@article{polo2024sloth,
  title={Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families},
  author={Maia Polo, Felipe and Somerstep, Seamus and Choshen, Leshem and Sun, Yuekai and Yurochkin, Mikhail},
  journal={arXiv preprint arXiv:2412.06540},
  year={2024}
}
```
