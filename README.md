# Machine Learning and Big Data - (KNN Models)

This repository showcases a machine learning group project completed as part of my **Machine Learning and Big Data** course at United States International University-Africa (USIU-Africa). The project focuses on building, hyperparameter tuning, and evaluating **K-Nearest Neighbors (KNN)** models for both classification and regression tasks using Python and scikit-learn.


## Project Overview & Datasets

The project applies the KNN algorithm across two distinct domains to analyze its performance, feature sensitivity, and limitations under different data distributions:

### 1. Part A: Medical Appointment No-Show (Classification)
* **Objective:** Predict whether a patient will miss their scheduled medical appointment ($1 = \text{Missed}, 0 = \text{Attended}$).
* **Dataset Scale:** 110,527 appointments across 14 variables.
* **Key Challenges:** Addressed severe class imbalance ($\sim 80/20$ split) and mixed numerical/categorical feature scales through rigorous preprocessing and threshold tuning.

### 2. Part B: Concrete Compressive Strength (Regression)
* **Objective:** Predict the continuous compressive strength of concrete in MPa based on raw material quantities and curing age.
* **Dataset Scale:** 1,030 samples with 8 ingredient inputs.
* **Key Challenges:** Handled bimodal zero-inflated ingredient distributions and non-linear, saturating effects of curing age.



## Technical Stack & Workflow

* **Language:** Python 3.x
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Core Methodologies:**
  * **Exploratory Data Analysis (EDA) & Cleaning:** Handled anomalous values (e.g., negative ages/wait times) and removed duplicate records to prevent data leakage.
  * **Feature Engineering:** Created meaningful derived metrics like `WaitDays` (lead time between booking and appointment).
  * **Robust Preprocessing:** Implemented `StandardScaler` fitted strictly on training subsets to prevent target/data leakage.
  * **Hyperparameter Optimization:** Utilized `GridSearchCV` to tune neighbor counts ($k$), distance weightings (`uniform` vs. `distance`), and distance metrics (`euclidean` vs. `manhattan`).
  * **Model Evaluation:** Assessed classification models using Precision, Recall, F1-score (minority class), and ROC-AUC; evaluated regression models via RMSE, MAE, and $R^2$ scores.

## Running the code
This analysis is contained within the Jupyter Notebook. You can view it directly on GitHub or run it locally by launching Jupyter.

