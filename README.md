<div align="center">

# Bayesian Hyperparameter Optimization in Deep Learning
### Loan Default Prediction · PyTorch + Optuna TPE

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-4.8.0-3E91CA?style=flat-square)](https://optuna.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle_PS_S4E8-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/playground-series-s4e8)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**Course:** Advanced Bayesian Inference for Data Science — DS630  
**Author:** Mostafa Hesham Mohamed Abdelzaher  
**Supervision:** Prof. Hegazy Zaher

</div>

---

## Overview

This project implements **Bayesian Hyperparameter Optimization (BHO)** to automatically tune a PyTorch multi-layer perceptron (MLP) for binary loan default prediction. Rather than exhaustively searching hyperparameter configurations via grid or random search, this work applies **Tree-structured Parzen Estimation (TPE)** — a Bayesian sequential model-based optimization algorithm — to intelligently navigate the hyperparameter space.

The central academic contribution is a grounded, code-traceable mapping of the three Bayesian components — **Prior**, **Likelihood**, and **Posterior** — onto concrete lines of the implementation. Each component is not merely theoretical; it has a direct computational role that can be pointed to in the codebase.

> **Key result:** Best validation **AUC = 0.9252** achieved in Trial 5 out of 50, with ~22 subsequent trials pruned by `MedianPruner` — demonstrating the efficiency gains of Bayesian sequential search over uninformed methods.

---

## Table of Contents

- [The Bayesian Framework](#the-bayesian-framework)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Key Engineering Decisions](#key-engineering-decisions)
- [References](#references)

---

## The Bayesian Framework

The optimization follows Bayes' theorem applied to the hyperparameter search space **θ**:

$$P(\theta \mid \mathcal{D}) \;\propto\; P(\mathcal{D} \mid \theta) \;\times\; P(\theta)$$

$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{Posterior}} \;\propto\; \underbrace{P(\text{AUC} \mid \theta)}_{\text{Likelihood}} \;\times\; \underbrace{P(\theta)}_{\text{Prior}}$$

Each component maps directly to the code:

---

### 1 · Prior — `P(θ)`

> *"What do we believe about good hyperparameters before seeing any results?"*

The prior is encoded in the **search space declarations** inside `objective()`. Every `trial.suggest_*` call defines a prior distribution over one hyperparameter:

```python
lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)   # log-uniform prior
n_layers     = trial.suggest_int("n_layers", 1, 4)                # discrete uniform
batch_size   = trial.suggest_categorical("batch_size", [256, 512, 1024])
weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
dropout      = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
```

The choice of `log=True` on `lr` and `weight_decay` is itself a **principled Bayesian prior decision**: it encodes the belief that *order of magnitude* matters more than absolute value — a jump from `1e-5 → 1e-4` is more meaningful than `9e-3 → 1e-2`. Trials 0–4 are sampled almost uniformly from these priors, before the surrogate model accumulates enough evidence to form a posterior.

---

### 2 · Likelihood — `P(AUC | θ)`

> *"Given that we tried configuration θ, how well did the model actually perform?"*

The likelihood is the **return value of `objective()`**. For each configuration θ, the function:
1. Builds the neural network with architecture dictated by θ
2. Trains it on the loan applicant data for `epochs` iterations
3. Evaluates AUC-ROC on a stratified held-out validation fold
4. Returns that score as the observed evidence

```python
auc = roc_auc_score(y_val, preds)
return auc   # ← this is the likelihood signal fed back to the surrogate
```

TPE then partitions all completed trials at a percentile threshold **γ = 0.25**:

| Group | Label | KDE Fitted |
|---|---|---|
| Top 25% by AUC | **Good** | `l(θ)` — density over high-performing configs |
| Bottom 75% by AUC | **Bad** | `g(θ)` — density over low-performing configs |

The **likelihood ratio** `l(θ) / g(θ)` serves as the acquisition function (Expected Improvement), measuring how much more likely a candidate θ is to fall in the good group.

---

### 3 · Posterior — `P(θ | AUC₁, AUC₂, …, AUCₙ)`

> *"After observing all trial results, what is our updated belief about where the optimum lies?"*

The posterior is the **continuously updated TPE surrogate model** — the pair `{l(θ), g(θ)}` refitted after every completed trial. The next configuration is chosen by:

$$\theta_{\text{next}} = \arg\max_{\theta} \;\frac{l(\theta)}{g(\theta)}$$

**Concrete evidence from the runs:** After Trial 5 (`lr=0.00188`, AUC=0.9252), the posterior over `lr` is no longer flat. All five high-AUC trials clustered in the range `lr ≈ 0.001–0.009`, causing the posterior to concentrate probability mass there. Trials 7 through 14 were subsequently **pruned by `MedianPruner`** — the posterior had already identified the promising region, and configurations outside it were terminated before completing full training.

The final **hyperparameter importance scores** (from `optuna.visualization.plot_param_importances`) are a direct readout of posterior sensitivity:

| Hyperparameter | Importance | Interpretation |
|---|---|---|
| `lr` | **0.34** | Posterior most peaked here — highest certainty |
| `epochs` | 0.27 | Second most informative dimension |
| `dropout_l0` | 0.18 | Regularization strength matters significantly |
| `n_layers` | 0.11 | Architecture depth has moderate influence |
| `n_units_l0` | 0.07 | Width less critical than depth for this dataset |
| `weight_decay` | 0.02 | Low sensitivity — L2 reg effect subsumed by dropout |
| `batch_size` | < 0.01 | Near-flat posterior — batch size barely matters |

---

## Dataset

**Source:** [Kaggle Playground Series S4E8 — Loan Default Prediction](https://www.kaggle.com/competitions/playground-series-s4e8)

| Property | Value |
|---|---|
| Rows | 58,645 |
| Raw features | 12 |
| Features after encoding | 20 |
| Target | `loan_status` (0 = repaid, 1 = default) |
| Task | Binary classification |

### Feature Engineering Pipeline

```
Raw CSV
  │
  ├── Drop: id
  ├── Binary encode: cb_person_default_on_file  (Y→1, N→0)
  ├── Ordinal encode: loan_grade                (A→1, B→2, C→3, D→4)
  ├── One-hot encode: person_home_ownership     (RENT/MORTGAGE/OWN/OTHER)
  ├── One-hot encode: loan_intent               (6 categories)
  ├── Log-transform: person_income              (range: 4,200 → 1,900,000)
  ├── Drop NaN rows                             (1,191 rows from loan_grade)
  └── StandardScaler on train split, transform on test
```

> **Note:** `data/` contains only a `.gitkeep` placeholder. Download `train.csv` from Kaggle and place it at `data/train.csv` before running.

---

## Project Structure

```
bayesian-hpo-loan-default/
│
├── README.md                          ← You are here
├── requirements.txt                   ← All dependencies pinned
├── .gitignore                         ← Excludes data, checkpoints, __pycache__
│
├── notebooks/
│   └── BHO_Loan_Default.ipynb         ← Main project notebook (full pipeline)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                  ← Cleaning, encoding, log-transform, scaling
│   ├── model.py                       ← create_model(trial, input_dim)
│   └── objective.py                   ← objective(trial) — the likelihood function
│
├── data/
│   └── .gitkeep                       ← Add train.csv here (not committed)
│
├── results/
│   ├── best_params.json               ← study.best_trial.params after run
│   └── plots/
│       ├── optimization_history.png   ← AUC per trial
│       └── param_importances.png      ← Posterior sensitivity summary
│
├── scripts/
│   └── save_results.py                ← Exports best_params.json + plots
│
└── assets/
    └── bayesian_cycle.png             ← Diagram used in README / slides
```

---

## Model Architecture

The network architecture is itself a hyperparameter — `create_model(trial, input_dim)` constructs a dynamic MLP where the number of layers, width, BatchNorm usage, and dropout rates are all suggested by the TPE sampler:

```
Input (20 features)
    │
    ├─ [Layer 1]  Linear(20 → n_units_l0)  →  [BatchNorm?]  →  ReLU  →  Dropout(p0)
    ├─ [Layer 2]  Linear(n_units_l0 → n_units_l1)  →  [BatchNorm?]  →  ReLU  →  Dropout(p1)
    ├─ [Layer n]  ...
    │
    └─ Linear(n_units_last → 1)   ← raw logit, no sigmoid
```

**Loss function:** `BCEWithLogitsLoss` (numerically stable, fuses sigmoid + BCE)  
**Optimizer:** `AdamW` with tunable `weight_decay`  
**Gradient clipping:** `max_norm=1.0` applied each step  
**Evaluation metric:** AUC-ROC on stratified validation fold

---

## Setup & Installation

**Requirements:** Python 3.10+, GPU optional (CPU is sufficient for this dataset)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/bayesian-hpo-loan-default.git
cd bayesian-hpo-loan-default

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the dataset
# Download train.csv from https://www.kaggle.com/competitions/playground-series-s4e8
# Place it at: data/train.csv
```

**`requirements.txt`**
```
torch>=2.0.0
optuna>=4.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
kaleido>=0.2.1
```

---

## How to Run

### Jupyter Notebook (recommended)

```bash
jupyter notebook notebooks/BHO_Loan_Default.ipynb
```

Run all cells in order. The study runs **50 trials** with a 1-hour `timeout` cap.

### Google Colab

Upload the notebook, enable GPU (T4), and upload `train.csv` to `/content/train.csv`.

### Saving results after the study

```python
import json, optuna.visualization as vis

# Save best hyperparameters
with open("results/best_params.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=2)

# Export plots (requires kaleido)
vis.plot_optimization_history(study).write_image("results/plots/optimization_history.png")
vis.plot_param_importances(study).write_image("results/plots/param_importances.png")
```

---

## Results

| Metric | Value |
|---|---|
| Best validation AUC | **0.9252** |
| Best trial | Trial 5 |
| Total trials | 50 |
| Pruned trials | ~22 |
| Sampler | `TPESampler(seed=42)` |
| Pruner | `MedianPruner(n_startup=5, warmup=3)` |
| GPU | NVIDIA T4 (Google Colab) |

**Best hyperparameter configuration (Trial 5):**

```json
{
  "lr": 0.001884,
  "batch_size": 256,
  "epochs": 16,
  "weight_decay": 0.000192,
  "n_layers": 3,
  "n_units_l0": 104,
  "use_bn_l0": false,
  "dropout_l0": 0.271,
  "n_units_l1": 8,
  "use_bn_l1": true,
  "dropout_l1": 0.355,
  "n_units_l2": 48,
  "use_bn_l2": false,
  "dropout_l2": 0.200
}
```

---

## Key Engineering Decisions

**`BCEWithLogitsLoss` over `BCELoss + Sigmoid`**  
Sigmoid saturates near 0 and 1, causing log(0) in the cross-entropy computation. `BCEWithLogitsLoss` fuses both operations into one numerically stable path using the log-sum-exp trick. Sigmoid is only applied explicitly at evaluation time for AUC computation.

**Log-transform on `person_income`**  
Raw income spans 4,200 → 1,900,000 (a 450× range). Applying `StandardScaler` directly creates extreme outlier z-scores that dominate early gradient updates. Log-transform compresses the distribution before scaling, ensuring income contributes proportionally to other features.

**`StratifiedKFold` for the validation split**  
The dataset is class-imbalanced (more `loan_status=0` than `1`). A random split risks a validation fold with too few defaults, inflating AUC estimates. Stratified splitting preserves the original class ratio in both fold partitions.

**`MedianPruner` with `n_warmup_steps=3`**  
Pruning before epoch 3 would discard trials that are still in the warm-up phase. After warmup, any trial whose AUC falls below the median of completed trials at the same epoch step is terminated — in this run, this saved approximately half of all trial compute.

**`log=True` on continuous hyperparameters**  
For learning rate and weight decay, the TPE sampler operates on the log-transformed domain. This is a direct prior choice: it assumes the objective function is smoother in log-space than in linear-space for these parameters, which aligns with standard deep learning practice.

---

## References

1. Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization*. Advances in Neural Information Processing Systems (NeurIPS), 24.

2. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*. Proceedings of the 25th ACM SIGKDD, 2623–2631.

3. Snoek, J., Larochelle, H., & Adams, R. P. (2012). *Practical Bayesian Optimization of Machine Learning Algorithms*. NeurIPS, 25.

4. AutoML.org. *HPO Overview: Expert-in-the-Loop HPO.* https://www.automl.org/hpo-overview/hpo-research/expert-in-the-loop-hpo/

5. Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). *Taking the Human Out of the Loop: A Review of Bayesian Optimization*. Proceedings of the IEEE, 104(1), 148–175.

6. Kaggle Playground Series S4E8 — Loan Default Prediction. https://www.kaggle.com/competitions/playground-series-s4e8

---

<div align="center">
<sub>DS630 · Advanced Bayesian Inference for Data Science · Mostafa Hesham Mohamed Abdelzaher</sub>
</div>
