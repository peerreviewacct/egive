# egive
Code repository for EGIVE (Efficient Global Interaction and Variable Explainability)
# 🔍 EGIVE — Efficient Global Interaction and Variable Explainability

> **A Fast, Model-Agnostic Framework for Global Interpretability of Black-Box Models**

---

## 📚 Publications

> **EGIVE: Efficient Global Interaction and Variable Explainability**  
> *Under review / Working paper*  
> **Authors:**   
> *(Update citation upon acceptance)*

---

## 📦 Overview

This repository provides an implementation of **EGIVE (Efficient Global Interaction and Variable Explainability)** —  
a fast, comprehensive, and **model-agnostic** framework for **global interpretability analysis** of black-box machine learning models.

While many interpretability tools focus on **local explanations** or rely on **model-specific assumptions**, EGIVE is designed for **global analysis**, characterizing:

- Single-variable effects  
- Pairwise interactions  
- User-defined three-way interactions  

across the **entire training distribution**, with **significantly reduced computational cost**.

EGIVE enables **interactive exploration** of variable importance and interaction structure, making it suitable for responsible ML, scientific discovery, and high-stakes decision-making domains such as healthcare.

---

## 🚀 Key Contributions

- ⚡ **Fast Global Interpretability:** Achieves orders-of-magnitude speedups over SHAP and interaction-based baselines.
- 🧩 **Unified Framework:** Computes feature importance, interaction strength, and partial dependence plots in a single pass.
- 🧠 **Model-Agnostic:** Applicable to Random Forests, Neural Networks, and arbitrary black-box predictors.
- 🔁 **Computation Reuse:** Reuses partial dependence evaluations to estimate interaction effects efficiently.
- 📊 **Comprehensive Outputs:** Supports single-feature effects, pairwise interactions, and selected three-way interactions.
- 🏥 **Real-World Impact:** Demonstrated on simulated benchmarks and real-world healthcare datasets.

---

## 🧠 Method Summary

EGIVE performs **global interpretability analysis** by combining:

- **Partial Dependence (PD)** for estimating marginal effects  
- **Inverse Propensity Weighting** for interaction estimation  
- **Efficient reuse of PD computations** to avoid redundant model evaluations  

### What EGIVE Computes

✔ Feature importance scores  
✔ Single-variable effects  
✔ Pairwise interaction strengths  
✔ User-specified three-way interactions  
✔ Partial dependence visualizations  

All within a **single unified workflow**.

---

## 🧪 Benchmark Results

EGIVE is benchmarked against **SHAP**, **sklearn permutation importance**, **$H^2$ interaction scores**, and **sklearn PDPs**.

### 🔹 Feature Importance Performance

- **Runtime:** Up to **30×–3000× faster** than SHAP  
- **Accuracy:** Correlation ≥ **0.89–0.99** with sklearn baselines  

### 🔹 Interaction Detection

- **AUC:** Up to **0.99** in identifying strong interactions  
- **Runtime:** Interaction scores computed at **zero additional cost**

### 🔹 Partial Dependence Accuracy

- **MAE:** As low as **0.02% of outcome standard deviation**
- **Runtime:** PD plots generated during feature importance computation

### 🔹 Total Runtime Comparison

| Model | EGIVE (s) | Benchmarks (s) |
|------|-----------|----------------|
| RF (continuous) | 53.9 | 87.5 |
| RF (binary) | 45.7 | 99.0 |
| NN (continuous) | 0.56 | 2.9 |
| NN (binary) | 1.27 | 4.2 |

> EGIVE consistently outperforms benchmark pipelines while providing **richer interpretability outputs**.

---

## 🧱 Framework Workflow

1. **Model Input**
   - Any trained black-box model (RF, NN, etc.)
   - Continuous or binary outcomes supported

2. **Global Sampling**
   - Uses training data distribution for global analysis

3. **Unified PD Computation**
   - Computes single-variable and interaction effects simultaneously

4. **Explainability Outputs**
   - Importance scores
   - Interaction rankings
   - Partial dependence plots

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/egive.git
cd egive
pip install -e .
pip install -r requirements.txt


## 🚀 Quick Start

```python
from egive import EGIVE

# Initialize EGIVE
explainer = EGIVE(
    model=trained_model,
    X_train=X_train,
    feature_names=feature_names
)

# Run global interpretability analysis
results = explainer.run(
    interactions="pairwise",      # or ["x1", "x2", "x3"] for three-way
    compute_pdp=True
)

# Access results
importance_scores = results.feature_importance
interaction_scores = results.interactions
pd_plots = results.partial_dependence

# Visualization
explainer.plot_importance()
explainer.plot_interactions(top_k=10)
explainer.plot_pdp(feature="age")
```

## 📊 Outputs

EGIVE returns:

- 📈 Feature importance rankings  
- 🔗 Interaction strength matrices  
- 📉 Partial dependence plots  
- 📁 Exportable results for downstream analysis  

All outputs are designed to be **interpretable, reproducible, and scalable**.

---

## 🧠 Applications

EGIVE is well-suited for:

- Healthcare analytics  
- Scientific modeling  
- Risk assessment  
- Policy evaluation  
- Responsible AI auditing  

---

## 📖 Citation

If you use EGIVE in your research, please cite:

```bibtex
@article{egive,
  title={EGIVE: Efficient Global Interaction and Variable Explainability},
  author={},
  journal={Under review},
  year={2026}
}

# eGIVE

> Interpretable Machine Learning Dashboard Generator

## Installation

```bash
pip install egive
```

## Quick Start

```python
from egive import run_egive

# Generate interpretability dashboard
run_egive(X, y, model, metric)
```

## Function Reference

### `run_egive()`

Generate a comprehensive dashboard of interpretable machine learning metrics for a trained model.

#### Syntax

```python
run_egive(X, y, model, metric, 
    predict_method=None, grid_size=20, h=200, w=200, barsize=10, fontsize=12,  feature_limit=None, pdp2_band_width=0.10, pdp_ips_trim_q=0.9, interaction_quantiles=(0.25, 0.75), twoway_to_threeway_ints=25,
    threeway_int_viz_limit=100, propensity_samples=1000, feature_imp_njobs=1, propensity_njobs=-1, pdp_legend=False, all_threeway_combinations=False
)
```

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `X` | | Tabular dataset of predictors. Accepts arrays or Pandas dataframes. |
| `y` | | Binary or continuous outcome vector, an array. |
| `model` | | Trained predictive model. Must have `predict` or `predict_proba` method for generating predictions. |
| `metric` | | Model performance metric for computing feature importances. Accepts `mae`, `mse`, `mae` for regressors, and `auc` for classifiers. Also accepts callable functions. If passing a function, higher values should represent poorer model performance. |

#### Optional Arguments

##### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `predict_method` | | `None` | Only used for binary classifier models. Set to `True` if feature importances should be computed using model's predict() method, as opposed to predict_proba(). If left as `None`, classifier importances will be computed with predict_proba() |

##### Visualization Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `grid_size` | int | `10` | Number of grid points for partial dependence functions. |
| `h` | int | `200` | Individual plot height, in pixels. |
| `w` | int | `200` | Individual plot width, in pixels. |
| `barsize` | int | `10` | Bar width, in pixels, for feature and interaction importances. |
| `fontsize` | int | `12` | Font size for plot labels. |
| `pdp_legend` | bool | `False` | Whether PDP plot should include a legend with variable labels. Recommended to leave as `False` unless multi-selecting PDPs for simultaneous visualization. |

##### Feature Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `feature_limit` | | `None` | Plots will only present importance and interaction scores for the top `feature_limit' most important features. |

##### Partial Dependence Plot (PDP) Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `pdp2_band_width` | float | `0.10` | Quantile bandwidth for computing pairwise interaction scores. |
| `pdp_ips_trim_q` | float | `0.9` | Quantile at which inverse propensity weights will be trimmed for multi-way partial dependence estimation. |

##### Interaction Analysis Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `interaction_quantiles` | tuple | `(0.25, 0.75)` | Quantiles used to define 'high' versus 'low' values of interacting variables, passed as an ordered tuple. 'Low' and 'high' partial dependence plots will be computed over rows where the interacting variable value is below the lower quantile and above the higher quantile. |
| `twoway_to_threeway_ints` | int | `25` | How many of the top-ranked pairwise interactions should be interacted with all features to generated candidate three-way interactions. For instance, in a dataset with `m` variables, each of the `m` variables will be interacted with the variable pairs from the top `twoway_to_threeway_ints` pairwise interactions, yielding `m` * `twoway_to_threeway_ints` candidate three-way interactions. |
| `threeway_int_viz_limit` | int | `100` | Number of highest-scoring three-way interactions for which three-way partial dependence plots should be included. Setting to `None` will allow all tested three-way interactions to be visualized with partial dependence plots, but will slow down the plot's rendering in the notebook console. |
| `all_threeway_combinations` | bool | `False` | Whether the `threeway_int_viz_limit` partial dependence visualizations should be used to visualize all possible combinations of the strongest interactions (`True`), or simply the `threeway_int_viz_limit` three-way partial dependence functions with the highest scores. |

##### Propensity Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `propensity_samples` | int | `1000` | Number of dataset samples used to estimate propensity scores for multi-way partial dependence functions.  |

##### Performance Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `feature_imp_njobs` | int | `1` | Number of cores (via `joblib`) to use when estimating univariate feature importances and partial dependence functions. |
| `propensity_njobs` | int | `-1` | Number of cores (via `joblib`) to use when computing propensity scores for multi-way partial dependence functions. |

#### Returns

[DESCRIBE WHAT THE FUNCTION RETURNS]

## Example Usage

```python
# Example with minimal arguments
from egive import run_egive
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Prepare data and model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Generate dashboard
dashboard = run_egive(
    X, y, model, 'auc',
    grid_size=10,
    feature_limit=5
)

# Print interactive dashboard to notebook console
dashboard
```



