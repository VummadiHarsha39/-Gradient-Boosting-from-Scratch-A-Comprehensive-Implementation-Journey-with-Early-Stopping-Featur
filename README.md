
A personal project to implement a Gradient Boosting Machine (GBM) from first principles, including custom decision trees, modular loss functions, early stopping, and feature importance.

# Gradient Boosting from Scratch: A Comprehensive Implementation Journey with Early Stopping & Feature Importance

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-gray?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-orange?style=for-the-badge&logo=matplotlib)
![Pandas](https://img.shields.io/badge/Pandas-lightgray?style=for-the-badge&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-orange?style=for-the-badge&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-orange?style=for-the-badge&logo=jupyter)
![Git](https://img.shields.io/badge/Git-red?style=for-the-badge&logo=git)

---

## Overview

This repository documents my experience in the development of a **Gradient Boosting Machine (GBM)** from scratch up in Python. The objective was to leave library calls and acquire a profound, intuitive knowledge of this potent ensemble algorithm, incorporating its mathematical underpinnings and practical implementation details. This project exhibits a dedication to practical machine learning features, a strong conceptual understanding, and modular software engineering.

## Features Implemented

My custom Gradient Boosting Machine is designed with modularity and real-world applicability in mind. It includes:

* **Custom Decision Tree (`tree.py`):**
    * A versatile base learner supporting both **regression** (minimizing MSE) and **classification** (minimizing Gini impurity).
    * Ability to intelligently handle datasets with both **numerical and categorical features**.
* **Modular Loss Functions (`loss.py`):**
    * Implementations for **Mean Squared Error (MSELoss)** for regression and **LogLoss** for binary classification.
    * Each loss function provides methods to calculate the `loss`, its `gradient`, and its `hessian`, crucial for the boosting process.
* **Core Gradient Boosting Loop (`booster.py`):**
    * The central orchestrator of the additive ensemble, iteratively fitting Decision Trees to negative gradients.
    * Initial prediction ($F_0$) logic (mean for regression, log-odds for classification).
* **Early Stopping (`early_stopping.py`):**
    * A vital mechanism that monitors validation loss during training.
    * Stops the boosting process prematurely if performance doesn't improve for a specified number of rounds (`patience`), preventing overfitting and saving computational resources.
* **Feature Importance Calculation:**
    * Quantifies each input feature's contribution to the overall model prediction.
    * Accumulates "gain" (impurity reduction) from splits across all trees in the ensemble and normalizes scores.
* **Well-Structured Repository & Demo Notebook (`notebooks/demo.ipynb`):**
    * A clean, organized codebase reflecting good software engineering practices.
    * A comprehensive Jupyter notebook showcasing all functionalities and results.

## Project Structure
/boosting_from_scratch_project/
├── boosting_from_scratch/
│   ├── init.py

│   ├── booster.py

│   ├── tree.py

│   ├── loss.py

│   ├── early_stopping.py

│   └── utils.py

│
├── notebooks/
│   └── demo.ipynb

│
├── README.md

└── requirements.txt


## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/boosting_from_scratch_project.git](https://github.com/YourGitHubUsername/boosting_from_scratch_project.git)
    cd boosting_from_scratch_project
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Install the project in editable mode (recommended for development):**
    This allows Python to recognize your local `boosting_from_scratch` package:
    ```bash
    pip install -e .
    ```

## Usage

Explore the `demo.ipynb` notebook to see the Gradient Boosting Machine in action and verify its components.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
2.  Navigate to the `notebooks/` directory and open `demo.ipynb`.
3.  **Run all cells** in the notebook.
    * The notebook will demonstrate:
        * Decision Tree Regressor & Classifier on numerical data.
        * Decision Tree on mixed numerical and categorical data.
        * Verification of MSELoss and LogLoss (gradients/Hessians).
        * GradientBooster for regression, classification, and mixed data.
        * Visualization of training/validation loss histories (from Early Stopping).
        * Plots of aggregated Feature Importances.



## Contributing

Feel free to fork this repository, explore the code, experiment, and suggest improvements!

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.
*(Note: You will need to create a `LICENSE.md` file in your root directory if you choose this. A simple MIT license file can be found online.)*

---
