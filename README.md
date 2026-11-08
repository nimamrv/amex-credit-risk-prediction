# American Express Credit Default Prediction

## Project overview

I built a machine learning pipeline to predict customer default risk from historical American Express account statements. The objective is to transform repeated monthly observations into a customer-level risk profile, train classification models, and evaluate how well those models distinguish customers who default from those who do not.

This project combines credit risk analytics, time-series feature engineering, gradient-boosted trees, neural networks, and model evaluation. It originated as an academic team project; this repository documents the pipeline and the work I can explain and demonstrate, without listing individual collaborators.

## Business problem

Credit decisions require a useful estimate of the probability that a customer will default. A customer's most recent statement alone may not capture the full picture: changes in payment behavior, balances, utilization, and other account characteristics over time can be informative. I therefore modeled default at the **customer level**, using each customer's sequence of monthly observations to summarize both longer-term behavior and recent changes.

The prediction target is binary: `target = 1` indicates default and `target = 0` indicates no default. I use **ROC-AUC** as the primary discrimination metric because the task involves ranking customers by risk across possible decision thresholds. AUC does not, by itself, establish that predicted probabilities are calibrated or that a particular lending decision is optimal.

## Data

**Source:** [American Express Default Prediction competition on Kaggle](https://www.kaggle.com/competitions/amex-default-prediction/data).

- `train_data.csv` contains multiple dated statement records per customer.
- `train_labels.csv` contains one binary target per `customer_ID`.
- `customer_ID` identifies the customer; `S_2` is the statement date.

The full raw data is not included in this repository. For the training command below, I use a **customer-history sample with labels already joined**. Each selected customer must retain their relevant monthly records, and the target is joined by `customer_ID`, not by row position. The current training entry point loads this prepared sample into memory; it is **not** a streaming loader for the full raw CSV.

## Pipeline design

### 1. Construct a consistent observation window

I first convert `S_2` to a date, sort statements by customer and date, and exclude statements after a configurable cutoff (`2018-03-31` by default). This ensures that each feature is computed using information available within the defined observation window.

I also check that the target is consistent across records belonging to the same customer. These checks are important because the final model expects exactly one label and one engineered feature row per customer.

### 2. Engineer customer-level features

Monthly records contain useful patterns that a single row cannot capture. In `src/features.py`, I build several groups of features:

| Feature group | How I calculate it | Why it matters |
|---|---|---|
| Overall numerical summaries | Mean, sum, minimum, maximum, standard deviation | Describes typical behavior, total activity, variability, and extremes |
| Recent activity windows | Mean, minimum, maximum, standard deviation over 3, 6, 9, and 12 months | Captures changes over different time horizons |
| Categorical behavior | One-hot category response rates and indicators of whether a category appeared in each lookback window | Converts repeated categorical states into customer-level signals |
| Recency | Days between the last statement and the cutoff date | Represents how recently a customer was observed |

These features are assembled into one row per `customer_ID`. Missing values are retained where an aggregation cannot be computed, and infinite values are converted to missing values. The XGBoost model can handle missing numerical feature values.

### 3. Split customers for model development

I use stratified, customer-level partitions:

- **70% training:** fit the feature selector and candidate models.
- **15% validation:** compare hyperparameter configurations.
- **15% held-out test:** evaluate the selected XGBoost approach once after model selection.

Because the split occurs **after** aggregation to one row per customer, a customer's monthly records cannot appear in multiple partitions. The split is randomized at the customer level, not a forward-in-time validation; assessing performance on future cohorts would require a separate temporal evaluation.

### 4. Select features and train XGBoost

XGBoost is my primary classifier because boosted decision trees are well suited to nonlinear relationships and interactions in tabular data.

I fit an initial XGBoost model on the training partition and retain features with importance greater than `0.005` (or the highest-ranked feature if none meet that threshold). I then search combinations of tree count, learning rate, row sampling, column sampling, and positive-class weighting, using **validation ROC-AUC** to select the configuration.

After selecting the hyperparameters, I fit the final XGBoost model on the combined training and validation partitions using the selected features, and calculate its ROC-AUC on the held-out test partition. I save the fitted model, selected feature names, and observation cutoff together in `outputs/xgb_model.joblib`.

### 5. Compare with a neural network (optional)

I also include an optional feed-forward neural network experiment. Unlike XGBoost, this model requires explicit preprocessing: I impute missing values and standardize the numerical inputs using statistics learned from the training partition only.

The network search varies hidden-layer count, nodes per layer, activation function, dropout, and batch size. I use binary cross-entropy loss and early stopping based on validation loss. The current implementation reports **validation ROC-AUC for neural-network configurations**; it does not yet produce a final held-out neural-network test result.

### 6. Evaluate and interpret the results

I compare models using ROC-AUC, which measures how well the model ranks positive cases above negative cases. For a credit risk use case, discrimination is only one part of the decision process. Before operational use, I would also examine calibration, decision thresholds, class-specific errors, stability over time, and potential fairness concerns.

The academic analysis also explored SHAP-based interpretation and illustrative credit approval strategies. Those analyses inform my discussion of model behavior and business decisions, but **SHAP plots and approval-strategy calculations are not executable outputs of this repository**.

## Results from the academic analysis

The academic project report recorded the following ROC-AUC scores:

| Model | Training ROC-AUC | Test set 1 ROC-AUC | Test set 2 ROC-AUC |
|---|---:|---:|---:|
| XGBoost | 0.950 | 0.945 | 0.942 |
| Neural network | 0.9419 | 0.9365 | 0.9319 |

XGBoost achieved the stronger ROC-AUC scores in that analysis. These are **results documented in the academic report**, not metrics generated by the current repository's training command. The repository uses its own train/validation/test evaluation protocol, so results from a new run may differ.

## Repository structure

```text
src/
  features.py          Customer-level feature engineering
  train.py             Data preparation, XGBoost search, evaluation,
                       and optional neural network comparison
scripts/
  make_synthetic.py    Generate fictional customer statement records
tests/
  test_features.py     Feature-engineering checks
  test_smoke.py        End-to-end test with synthetic data
requirements.txt       Python dependencies
.gitignore             Excludes datasets, model artifacts, and local files
README.md              Project methodology and usage
```

## How to run

Use Python in a terminal from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Generate a small **synthetic dataset** and run a quick end-to-end check:

```powershell
python scripts/make_synthetic.py
python -m src.train --data data/synthetic_customers.csv --quick
```

To train on a prepared real-data sample that already contains `customer_ID`, `S_2`, `target`, and numerical statement features:

```powershell
python -m src.train --data data/sampled_train_data.csv --quick
```

Remove `--quick` to evaluate the full XGBoost hyperparameter grid. To run the optional neural network search, install a compatible TensorFlow version and add `--run-nn`.

The training command writes its results to `outputs/`, including `metrics.json`, the XGBoost validation grid, and the saved XGBoost model. The `--quick` flag limits the number of configurations and is intended for execution checks, not comprehensive tuning.

**Synthetic data is for software testing only.** Its values are fictional, and any metrics produced from it should not be interpreted as credit risk performance.

## Practical considerations

This repository demonstrates a modeling workflow, not a production lending system. The current implementation has been checked with unit tests and a synthetic-data smoke test, but it has **not been run end to end on the original 15 GB training dataset**. Running on the raw Kaggle files requires preparing a manageable customer-level sample and joining labels first.

For a production setting, I would extend the work with temporal out-of-sample testing, probability calibration, bias and fairness assessments, drift monitoring, stronger data validation, and a documented decision policy. I would also validate the business assumptions behind any proposed approval thresholds before recommending them for use.
