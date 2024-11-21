# Optimized Prediction of Weapon Effectiveness in BVR Air Combat Scenarios Using Enhanced Regression Models

This repository contains the code, data, and implementation details for the paper:

```bibtex
@article{kuroswiski4WEZ,
  title={Optimized Prediction of Weapon Effectiveness in BVR Air Combat Scenarios Using Enhanced Regression Models},
  author={Kuroswiski, Andre R. and Wu, Annie S. and Passaro, Angelo},  
}
```
Status: *In Revision*

## Overview
This project provides the implementation for predicting the **Weapon Engagement Zone (WEZ)** in **Beyond-Visual-Range (BVR)** air combat scenarios, based on the methods and results presented in the referenced paper. Accurate WEZ predictions are critical for decision-making in air combat and the development of efficient autonomous systems.

### Key Contributions
- **Modeling Approach**: Comparison of multiple regression methods, including Lasso, Ridge, Polynomial Regression (PR), Multi-Layer Perceptrons (MLP), and others, emphasizing the potential of PR-based solutions with regularization.
- **Feature Engineering**: Achieved up to **70% improvement** in Mean Absolute Error (MAE) through feature engineering and data augmentation.
- **Efficiency**:  Demonstrated that Lasso regression with higher interaction degrees (12th) can be **33% better in accuracy** and is **2.1 times faster**, than more complex models like MLP.
- **Simplification**: Simplified PR-based models maintain high accuracy while significantly reducing prediction times, supporting real-time and accelerated simulations.
- **Open Dataset**: Includes a dataset from high-fidelity air combat simulations to promote reproducibility and further research.

## Objectives
- Develop and evaluate WEZ prediction models using experimental data.
- Compare regression methods for accuracy, computational efficiency, and portability.
- Implement innovative preprocessing techniques to enhance learning.

## Usage
### Requirements
- Python 3.8+
- Libraries: `scikit-learn`, `numpy`, `pandas`

## Dataset
This repository includes a dataset generated using the **Aerospace Simulation Environment (ASA)** (https://asa.dcta.mil.br/) . The dataset represents air combat scenarios with features such as aircraft speed, altitude, and heading, and outputs like WEZ ranges (`R_max` and `R_nez`).

---
