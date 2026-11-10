# Red and White Wine Quality Modelling

CSCI323 Modern AI — Group FT13
Yong Le, Jiachen, Jing Ting,Jae Hyun, Ian and Sherya.

## Project Overview

This project investigates how well physicochemical measurements can predict wine quality. We train and evaluate regression models separately on red and white wine datasets and compare their performance.

Our aim is to understand how different methods, model settings and data characteristics affect predictions, including errors on rare low and high quality scores.

## Dataset

We use the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality), containing 11 physicochemical input features and a sensory quality score as the target.

| Dataset | Original records | Duplicates removed | Records used | Training | Testing |
|---|---:|---:|---:|---:|---:|
| Red wine | 1,599 | 240 | 1,359 | 1,087 | 272 |
| White wine | 4,898 | 937 | 3,961 | 3,168 | 793 |

Exact duplicate rows are removed before splitting. Red and white wine are processed and modelled separately.

## Methods

We compare six main regression methods:

- Linear Regression
- Decision Tree
- Random Forest
- Multi-Layer Perceptron (MLP)
- Gradient Boosting
- K-Nearest Neighbours (KNN)

XGBoost is included as an additional exploratory comparison. Its settings are adapted from the Gradient Boosting search rather than selected through a separate parameter search.

## Experimental Procedure

- Use an 80/20 train-test split, stratified by quality score, with random seed 42.
- Keep the same split for all models within each dataset.
- Evaluate and select configurations using shuffled five-fold cross-validation on training data.
- Fit scaling for MLP and KNN within training pipelines.
- Investigate model settings, training-data size and feature selection.
- Examine errors at individual quality scores and within broader quality groups.

Performance is assessed using:

| Metric | Interpretation | Better |
|---|---|---|
| MAE | Average absolute error in quality-score points | Lower |
| RMSE | Error measure that penalises larger mistakes more strongly | Lower |
| R² | Performance relative to predicting the mean of the evaluated targets | Higher |

## Notebook Contents

The notebook includes:

1. Exploratory data analysis and cleaning
2. Dataset splitting and preparation
3. Baseline model implementations
4. Cross-validation and baseline comparisons
5. Hyperparameter and architecture experiments
6. Training-data sensitivity and feature selection
7. Marginal-case analysis and feature importance
8. Supplementary XGBoost comparison
9. Complete red and white wine model comparison

## Additional Notebooks

- `RF_Regression_vs_Classification.ipynb`: Random Forest regression vs classification, error by quality score, and feature importance of the tuned Random Forest. Uses the same cleaning and split as the main notebook.
- `Log_Transform_and_Feature_Subsets.ipynb`: log transformation of skewed features, and feature subsets compared with all 11 features.
- `Rare_Wine_Weighting_oversampling.ipynb`: class weighting and oversampling for rare quality scores (3-4 and 8-9) on the four selected models, comparing overall error and rare wine error. Uses the same cleaning, split and tuned models as the main notebook.

## Reading the Outputs

Red and white calculations are kept separate, with combined tables and figures provided for comparison. Many individual calculation cells intentionally suppress output.

Section 9 contains the complete main-model comparison. The supplementary XGBoost results appear in Section 8.5.6.

## Limitations

- Rare quality scores have few test observations, so extreme-score findings should be interpreted cautiously.
- Test results were inspected during development; comparisons are exploratory rather than a fresh confirmatory evaluation.
- Cross-validation fold standard deviations do not establish statistical significance.
- XGBoost’s round count is selected before its subsequent cross-validation, so that CV score does not independently evaluate its selection procedure.
- Feature importance describes model behaviour and does not establish causal relationships.
- Prediction accuracy is limited to the information available in the chemical measurements.

## Research Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

The article motivates the wine-quality regression problem. Our model configurations and evaluation procedure differ from the original study.



