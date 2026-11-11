# Wine Quality Modelling and Preprocessing Experiments

**CSCI323 Modern AI — Group FT13**

## Overview

This project investigates how preprocessing and imbalance-handling strategies affect wine-quality prediction.

Using 11 physicochemical measurements, we predict numerical quality scores for red and white wines separately. We compare four models and examine how changes to their inputs and training data affect overall accuracy and predictions for rare quality scores.

## Dataset

We use the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality).

| Dataset | Original records | Duplicates removed | After cleaning | Training | Testing |
|---|---:|---:|---:|---:|---:|
| Red wine | 1,599 | 240 | 1,359 | 1,087 | 272 |
| White wine | 4,898 | 937 | 3,961 | 3,168 | 793 |

Exact duplicates are removed before splitting. Each dataset uses a stratified 80/20 train-test split with random seed 42.

## Models

- Linear Regression
- Random Forest
- K-Nearest Neighbours (KNN)
- Multi-Layer Perceptron (MLP)

The main experiments keep previously selected model settings fixed to investigate the effects of preprocessing.

## Experiments

| Experiment | Models |
|---|---|
| Original-distribution reference | All four |
| No scaling, standard scaling and robust scaling | KNN and MLP |
| Log transformation of skewed inputs | All four |
| Selection of 3, 5, 8 or all 11 features | All four |
| Random oversampling | All four |
| Soft and full inverse-frequency sample weighting | Linear Regression and Random Forest |
| Separately tuned regression versus classification | Random Forest |
| Permutation and impurity feature importance | Random Forest regressor |

Preprocessing and feature selection are fitted within training partitions. Oversampling and sample-weight calculation use training observations only. Validation and test distributions remain unchanged.

For MLP, the early-stopping validation set is separated before preprocessing and oversampling.

## Evaluation

The experiments share the same five-fold cross-validation partitions within each dataset.

We report:

- Overall MAE, RMSE and R².
- Macro MAE: the equally weighted mean of per-score MAEs.
- MAE and sample count at each quality score.
- Rare-score MAE for scores ≤4 or ≥8.
- Typical-score MAE for scores 5–6, with score 7 reported separately.

Cross-validation metrics are calculated from combined out-of-fold predictions. The RF regression/classification experiment also reports exact and within-one-score accuracy.

## Main Notebook

**`Wine_Quality_Final.ipynb`**

The notebook presents each experiment section by section:

1. Data loading, cleaning and splitting.
2. Shared preprocessing and evaluation rules.
3. Four reference models.
4. Reference-model comparison.
5. Scaling experiments.
6. Log-transformation experiments.
7. Feature-subset experiments.
8. Rare-score weighting and oversampling.
9. RF regression versus classification.
10. RF feature importance.
11. Consolidated results and downloadable outputs.

Each main experiment includes red-wine calculations, white-wine calculations, and comparison tables and graphs.

## Baseline and Tuning Notebook

**`Red_White_Wine_Quality_Modelling_Comparisons.ipynb`**

This earlier notebook contains the baseline models and hyperparameter searches (Random Forest grid search, KNN grid search and MLP architecture comparison) that produced the selected settings used in the main notebook. The baseline and selected-configuration results in Sections 3.2 and 3.3.1 of the report come from this notebook.

## Running in Google Colab

1. Upload `Wine_Quality_Final.ipynb` to Google Colab.
2. Connect to a Python runtime.
3. Check the configuration:

   ```python
   QUICK_MODE = False
   RUN_HOLDOUT = True
   RUN_RF_GRID_SEARCH = True
   ```

4. Select **Runtime → Run all**.
5. Save the executed notebook.

An internet connection is required to download the datasets unless local copies are available. Missing core packages are installed by the setup cell. Full execution, particularly RF grid searches and MLP experiments, can take considerable time.

`QUICK_MODE = True` is intended only to check execution. Its results should not be used as final experimental evidence.

## Outputs

Results are saved in `wine_section_outputs/`, including:

- Cross-validation and exploratory test metrics.
- Out-of-fold and test predictions.
- Per-quality-score errors.
- Oversampling counts and duplication factors.
- Fold-specific preprocessing information.
- Comparison figures.
- Software versions and experiment settings.

The final cell creates `wine_section_outputs.zip` for download.

## Limitations

- Rare extreme scores have few independent observations.
- Oversampling repeats existing records rather than creating new information.
- Improvements in rare-score error can come at the cost of overall accuracy.
- Previously selected model settings make the CV comparisons conditional rather than nested evaluations of the complete selection process.
- The test set was inspected during development, so holdout comparisons are exploratory.
- Single-seed results and fold standard deviations do not establish statistical significance.
- Feature importance describes model reliance and does not demonstrate causation.

## Research Reference

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems, 47*(4), 547–553. https://doi.org/10.1016/j.dss.2009.05.016

The article supports the wine-quality regression problem. Our implementations and evaluation procedure differ from the original study. Additional method references are provided in the notebook and report.

## Contributors

Group FT13: Yong Le, Jae Hyun, Ian, Jiachen, Jing Ting and Shreya.
