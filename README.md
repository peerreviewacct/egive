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
@article{aslani_egive,
  title={EGIVE: Efficient Global Interaction and Variable Explainability},
  author={},
  journal={Under review},
  year={2026}
}

