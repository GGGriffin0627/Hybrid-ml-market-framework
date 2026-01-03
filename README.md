# Hybrid Machine Learning and Numerical Optimization Framework
for Financial Market Prediction

This repository implements a hybrid framework for financial market prediction that integrates
machine learning models with numerical optimization for direct portfolio construction.

The framework was developed for the **Hull Tactical Market Prediction Challenge**, where the task
is reframed from return forecasting to volatility-constrained position optimization.

---

## Motivation

In quantitative trading, strong predictive signals do not necessarily translate into good trading
performance once practical constraints such as volatility limits and position bounds are imposed.
This project addresses the gap between **prediction accuracy** and **actionable portfolio construction**
by explicitly optimizing the evaluation objective.

---

## Framework Overview

The pipeline consists of two main stages:

1. **Prediction Stage**
   - Two LightGBM models with complementary hyperparameters
   - A simplified Transformer for sequential dependency modeling
   - Ridge regression used as a meta-learner for model blending

2. **Optimization Stage**
   - Portfolio weights are optimized directly using Powell’s method
   - Objective aligns with the scoring function
   - Explicit constraints on volatility and position bounds

This design ensures that model outputs are aligned with real-world evaluation criteria rather than
proxy loss functions.

---

## Repository Structure

```text
src/
├── features/        # Feature engineering and preprocessing
├── models/          # LightGBM and Transformer implementations
├── optimization/    # Numerical optimization routines
├── evaluation/      # Validation and scoring utilities
├── inference/       # Kaggle inference server interface

