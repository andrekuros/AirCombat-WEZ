# Optimized Prediction of Weapon Effectiveness in BVR Air Combat Scenarios Using Enhanced Regression Models

This repository contains the code, data, and implementation details for the paper:

```bibtex
@ARTICLE{kuroswiski2025wez,
  author={Kuroswiski, Andre R. and Wu, Annie S. and Passaro, Angelo},
  journal={IEEE Access}, 
  title={Optimized Prediction of Weapon Effectiveness in BVR Air Combat Scenarios Using Enhanced Regression Models}, 
  year={2025},
  volume={13},  
  pages={21759-21772},
  doi={10.1109/ACCESS.2025.3535555}}

```

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


### Repository Structure

This repository is organized as follows:

- `Data/`: Contains all the input datasets necessary for running the experiments. 
- `Output/`: Stores all generated outputs created during the execution.
- `Paper_Results/`: Contains the results referenced in the paper.
- `WEZ_Model_Generation.ipynb`: The main notebook for executing the experiments and generating the results.
- `requirements.txt`: A list of Python dependencies required to run the notebook.
- `README.md`: Documentation for the project.
- `LICENSE`: Licensing information for the project.

---

### Running the Code

To run the experiments, follow these steps:

1. **Set Up Environment**:
   Ensure you have Python 3.8+ installed. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

2. **Prepare the Data**: Place the necessary datasets in the Data/ folder. The datasets should follow the expected format outlined in the notebook.

3. **Execute the Notebook**: Open the WEZ_Model_Generation.ipynb notebook in a Jupyter environment. Follow the steps outlined in the notebook to:
    - Preprocess the data.
    - Train the models.
    - Evaluate the results.

4. **Check the Outputs**: The generated results will be stored in the Output/ directory.

---

# Dataset Description

The dataset used in this study was generated using high-fidelity air combat simulations to support the development and evaluation of Weapon Engagement Zone (WEZ) prediction models. The dataset captures various scenarios and conditions, representing the engagement between a shooter aircraft and a target.

### Features
The dataset includes key input features that influence missile performance, derived from both raw simulation parameters and feature engineering processes. Below is a table detailing these features:

| **Feature**                     | **Variable**       | **Min Value** | **Max Value** | **Unit**        | **Description**                                                                                      |
|----------------------------------|--------------------|---------------|---------------|-----------------|------------------------------------------------------------------------------------------------------|
| Shooter Speed                   | `v_s`             | 450           | 750           | NM/hour         | Shooter aircraft's ground speed.                                                                    |
| Shooter Altitude                | `h_s`             | 1,000         | 45,000        | ft              | Shooter aircraft's altitude above sea level.                                                        |
| Target Speed                    | `v_t`             | 450           | 750           | NM/hour         | Target aircraft's ground speed.                                                                     |
| Target Radial                   | `φ_t`             | -60           | 60            | degree          | Angular position of the target relative to the shooter.                                             |
| Target Altitude Difference      | `Δh_t`            | -5,000        | 5,000         | ft              | Difference in altitude between the shooter and target (`h_s - h_t`).                                |
| Relative Heading to Radial      | `Δφ_t`            | -180          | 180           | degree          | Relative angle combining the target's radial and headings (`θ_t - θ_s - φ_t`).                      |


### Target Outputs
The dataset contains two primary outputs for each engagement scenario:
- **Maximum Range for Weapon Engagement Zone (`R_max`)**: The furthest distance at which a missile can successfully engage the target.
- **No Escape Zone (`R_nez`)**: The range within which the target cannot evade the missile, regardless of maneuvers.

### Data Generation Process
1. **Simulation Environment**:
   - The simulations were conducted using the Aerospace Simulation Environment (ASA), a high-fidelity platform based on MIXR.
   - Scenarios were designed to include realistic conditions for air combat, ensuring diverse and representative data.

2. **Binary Search for Outputs**:
   - The `R_max` and `R_nez` values were determined through iterative binary searches. Initial ranges started at 45 NM, and simulations refined these values with an accuracy threshold of 0.2 NM.

3. **Experiment Design**:
   - Two datasets were created:
     - A **factorial design dataset** with a fixed set of input levels for initial feature analysis (864 cases).
     - A **random design dataset** with 1,000 cases generated using uniformly random input values, designed for model training and evaluation.

---
