# Credit Card Fraud Detection

This project uses machine learning to classify credit card transactions as normal or fraudulent.

The project covers the complete workflow, from inspecting and cleaning the dataset to exploring class imbalance, training models, evaluating their performance, and saving the final model for future predictions.

## Project Overview

The dataset contains a very small number of fraudulent transactions compared with normal transactions, so accuracy alone is not enough to evaluate the models.

The project covers:

- Dataset inspection
- Data cleaning
- Exploratory data analysis
- Class imbalance analysis
- Train/validation/test split
- Random Forest and XGBoost
- Precision, recall, F1-score and ROC-AUC
- Confusion matrix
- Threshold analysis
- Saving and loading the final model

## Dataset

The dataset contains **284,807 transactions and 31 columns**.

`Class` is the target column:

- `0` = normal transaction
- `1` = fraudulent transaction

The original dataset contains:

- 284,315 normal transactions
- 492 fraudulent transactions
- 1,081 exact duplicate rows
- No missing values

After removing exact duplicate rows, the dataset contains:

- 283,726 transactions
- 283,253 normal transactions
- 473 fraudulent transactions

The features are:

- `Time`
- `Amount`
- `V1` through `V28`

The meanings of the `V1` to `V28` features are not available in the supplied dataset, so they are treated as numerical features without assigning additional meanings to them.

### Original Dataset

The original CSV is not included in this GitHub repository because it is around **143 MB**, which is larger than GitHub's normal file size limit.

Dataset: **[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud]**

The cleaned dataset generated during preprocessing is stored as:

```text
data/cleaned_data.csv.gz
```

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── cleaned_data.csv.gz
│
├── models/
│   └── fraud_detection_pipeline.joblib
│
├── notebooks/
│   ├── inspection.ipynb
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   ├── feature_engineering.ipynb
│   └── model.ipynb
│
├── README.md
└── requirements.txt
```

## Workflow

The project follows this workflow:

```text
Dataset
   ↓
Inspection
   ↓
Preprocessing
   ↓
EDA
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
   ↓
Threshold Analysis
   ↓
Final Model
   ↓
Saved Pipeline
```

## Notebooks

### 1. inspection.ipynb

This notebook checks the original dataset before making any changes.

It covers:

- Dataset shape
- Column names
- Data types
- Missing values
- Duplicate rows
- Target values
- Class distribution
- Basic statistics
- Example normal and fraud transactions

### 2. preprocessing.ipynb

This notebook cleans the dataset and removes exact duplicate rows.

The cleaned dataset is saved as:

```text
data/cleaned_data.csv.gz
```

### 3. eda.ipynb

This notebook explores the cleaned dataset and focuses mainly on:

- Class imbalance
- Fraud vs normal transactions
- Feature distributions
- Transaction amount
- Correlations
- Differences between the two classes

### 4. feature_engineering.ipynb

This notebook separates the features and target and creates the train/validation/test split.

The split is stratified so that the fraud and normal transaction ratio is maintained across the datasets.

### 5. model.ipynb

This notebook trains and evaluates Random Forest and XGBoost.

It also includes:

- Confusion matrices
- Classification reports
- Model comparison
- Threshold analysis
- Final model selection
- Saving and loading the final pipeline

## Class Imbalance

Fraud is a very small part of the dataset.

Before removing duplicates, only **492 out of 284,807 transactions** were fraudulent, which is about **0.173%**.

This means that a model could achieve very high accuracy simply by predicting most transactions as normal.

Because of this, the project focuses more on precision, recall, F1-score and ROC-AUC rather than accuracy alone.

The models handle the imbalance using class weighting rather than applying resampling to the validation or test data.

- Random Forest uses balanced class weights.
- XGBoost uses a positive-class weight calculated from the training data.

## Models Used

### Random Forest

Random Forest is a tree-based classification model that combines multiple decision trees.

Class weighting is used to give more importance to the minority fraud class.

### XGBoost

XGBoost is another tree-based model used for comparison.

The positive-class weight is calculated from the training data to account for the imbalance between normal and fraudulent transactions.

Neither model requires feature scaling for this dataset.

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix
- Classification Report

For fraud detection, precision and recall are particularly useful.

**Recall** tells us how much of the actual fraud the model is able to detect.

**Precision** tells us how many of the transactions flagged as fraud are actually fraudulent.

There is a tradeoff between the two, so the choice of threshold depends on how the cost of missed fraud compares with the cost of false alarms.

## Results

The following results are measured on the validation split using a classification threshold of `0.50`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.999692 | 0.969697 | 0.842105 | 0.901408 | 0.958803 | 0.882258 |
| XGBoost | 0.998877 | 0.612613 | 0.894737 | 0.727273 | 0.968637 | 0.876328 |

Random Forest had the higher validation F1-score and was selected as the final model.

### Final Test Results

The selected Random Forest model was then evaluated on the untouched test split using the selected `0.50` threshold.

| Metric | Result |
|---|---:|
| Accuracy | 0.999489 |
| Precision | 0.958333 |
| Recall | 0.726316 |
| F1-score | 0.826347 |
| ROC-AUC | 0.938925 |
| Average Precision | 0.810457 |

The confusion matrix was:

```text
[[56648,     3],
 [   26,    69]]
```

Rows represent the actual class and columns represent the predicted class.

This means that in this test split:

- 56,648 normal transactions were correctly classified.
- 3 normal transactions were flagged as fraud.
- 69 fraudulent transactions were correctly detected.
- 26 fraudulent transactions were missed.

## Threshold Analysis

The default classification threshold of `0.50` was also compared with other thresholds using the validation data.

For Random Forest:

| Threshold | Precision | Recall |
|---:|---:|---:|
| 0.30 | 0.915 | 0.855 |
| 0.50 | 0.970 | 0.842 |
| 0.70 | 0.982 | 0.737 |

Lowering the threshold makes the model more willing to classify a transaction as fraud. In this case, recall increases slightly while precision decreases.

Increasing the threshold has the opposite effect. Fewer transactions are flagged as fraud, which increases precision but reduces recall.

The appropriate threshold would ultimately depend on the actual cost of missed fraud and false alarms.

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Open the project directory and start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in this order:

1. `inspection.ipynb`
2. `preprocessing.ipynb`
3. `eda.ipynb`
4. `feature_engineering.ipynb`
5. `model.ipynb`

The original CSV is not stored inside the repository.

If you want to run the first two notebooks from scratch, download the original dataset from the dataset link above and update the CSV path in the notebooks to point to your local copy.

The later notebooks use the cleaned dataset generated during preprocessing.

## Saved Model

The final model is saved as:

```text
models/fraud_detection_pipeline.joblib
```

The saved pipeline can be loaded later using `joblib`.

Example:

```python
import joblib

model = joblib.load("models/fraud_detection_pipeline.joblib")

prediction = model.predict(new_data)
fraud_probability = model.predict_proba(new_data)[:, 1]
```

`predict()` uses the model's default `0.50` classification threshold.

The threshold analysis in the model notebook shows how this can be changed when a different operating point is required.

New data must contain the same 30 feature columns used by the model and follow the same structure as the training data.

## Requirements

The main libraries used in the project are:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib

The exact dependencies are listed in:

```text
requirements.txt
```

## Future Improvements

Some possible improvements for a future version are:

- Select the operating threshold using the actual cost of missed fraud and false alarms.
- Test the model on newer transaction data.
- Perform additional model tuning.
- Monitor model performance over time after deployment.