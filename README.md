# Customer Spending Score Prediction using Ensemble Learning

An end-to-end Machine Learning pipeline that predicts customer `Spending_Score` tiers based on demographic, professional, and familial characteristics. This repository demonstrates automated data preprocessing, advanced feature engineering, target/feature encoding, stratified train-test splitting, feature standardization, and an ensemble modeling strategy combining **Random Forest**, **Extra Trees**, **XGBoost**, and **LightGBM** via **Hard** and **Soft Voting** classifiers across three Jupyter Notebooks.

---

## 📌 Project Overview

This project streams customer data from Hugging Face, performs automated cleaning and missing value imputation, engineers domain-specific interaction features, normalizes numeric attributes, and trains ensemble classifiers to predict multi-class spending score categories.

### Key Highlights

* **Automated Data Retrieval**: Streams `jason1966/abisheksudarshan_customer-segmentation` via the Hugging Face `datasets` library.
* **Modular Notebook Structure**: Separates exploratory analysis (`EDA.ipynb`), model/hyperparameter experimentation (`models.ipynb`), and the end-to-end pipeline execution (`main.ipynb`).
* **Feature Engineering Pipeline**: Generates interaction terms, ratios, and categorical binnings (e.g., age groups, career stages, combined marital-profession attributes).
* **Hybrid Ensemble Architecture**: Merges bagging (**Random Forest**, **Extra Trees**) and boosting (**XGBoost**, **LightGBM**) algorithms.
* **Meta-Voting Classifiers**: Evaluates both **Hard** (majority vote) and **Soft** (averaged class probabilities) voting strategies.
* **Evaluation & Visualization**: Outputs performance metrics (precision, recall, F1-score) and side-by-side Seaborn confusion matrix heatmaps.

---

## 📂 Repository Structure

```text
.
├── EDA.ipynb        # Exploratory Data Analysis, missing value evaluation, and distribution plots
├── models.ipynb     # Model experimentation, baseline comparisons, and hyperparameter testing
├── main.ipynb       # End-to-end execution pipeline (Ingestion -> Preprocessing -> Ensembling)
└── README.md        # Project documentation

```

### Notebook Workflow Overview

* **`EDA.ipynb`**: Focuses on data inspection, visual analysis of demographic trends, target distribution, and identifying missing value patterns.
* **`models.ipynb`**: Used for benchmarking base estimators (`RandomForestClassifier`, `ExtraTreesClassifier`, `XGBClassifier`, `LGBMClassifier`), experimenting with feature interaction sets, and testing individual hyperparameter configurations.
* **`main.ipynb`**: Integrates the complete production workflow. Loads the raw dataset, runs preprocessing, builds the voting ensembles, and generates evaluation reports alongside confusion matrix visualizations.

---

## 🛠️ Data & Feature Pipeline

### 1. Data Ingestion & Preprocessing

* **Dataset Loading**: Imports core packages (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`) and loads the dataset from Hugging Face into a Pandas DataFrame. Column headers are automatically stripped of leading and trailing whitespaces.
* **Missing Value Imputation**:
* Categorical features (`Ever_Married`, `Graduated`, `Profession`) fill missing values with `'Unknown'`.
* Numerical features (`Work_Experience`, `Family_Size`) impute missing entries using their median values.



### 2. Feature Engineering & Interaction Terms

* **Continuous Ratios & Flags**:
* `Experience_Age_Ratio`: Work experience divided by customer age.
* `Is_Married`, `Graduated_Int`: Binary representations of demographic flags.
* `Age_Married_Interact`, `Age_Graduated_Interact`: Continuous interaction terms measuring age against marital and education status.


* **Composite Categoricals**:
* `Married_Profession`: Composite feature combining marital status and profession.


* **Household Density**:
* `Age_Per_Family_Member`: Dynamically scales customer age by family size when `Family_Size` is present.


* **Binning & Discretization**:
* `Age_Group`: Binned into 4 life stages (*Young*, *Adult*, *Mid_Age*, *Senior*).
* `Career_Stage`: Discretizes work experience into career levels (*Junior*, *Mid*, *Senior*, *Veteran*).



### 3. Encoding, Splitting & Scaling

* **Feature & Target Separation**: Identifies and drops identifier and target columns (`Spending_Score`, `Customer_ID`, `ID`) to isolate the feature matrix $X$, preserving `Spending_Score` as the raw target label $y_{\text{raw}}$.
* **Label & One-Hot Encoding**:
* Fits a `LabelEncoder` on the multi-class target variable ($y$).
* Transforms all remaining string and categorical feature columns into binary dummy variables using `pd.get_dummies(..., drop_first=True)` to prevent multicollinearity.


* **Stratified Splitting & Standardization**:
* Splices the data into an **80/20** train-test split with class stratification (`stratify=y`) to maintain target class balance.
* Standardizes all numerical features using `StandardScaler` (fitted strictly on $X_{\text{train}}$ to prevent data leakage).



---

## 🤖 Model Architecture & Technical Details

The pipeline uses four distinct algorithms to balance model variance and bias before passing predictions to meta-voting classifiers.

```text
                       ┌─────────────────────────┐
                       │   Input Features (X)    │
                       └────────────┬────────────┘
                                    │
           ┌────────────────┬───────┴────────┬────────────────┐
           ▼                ▼                ▼                ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │ Random      │  │ Extra       │  │ XGBoost     │  │ LightGBM    │
    │ Forest      │  │ Trees       │  │             │  │             │
    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
           │                │                │                │
           └────────────────┼────────────────┴────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
      ┌─────────────────┐       ┌─────────────────┐
      │  Hard Voting    │       │  Soft Voting    │
      │  (Majority Vote)│       │  (Probability)  │
      └─────────────────┘       └─────────────────┘

```

### 1. Bagging Models (Parallel Ensembles)

* **Random Forest (`RandomForestClassifier`)**: Builds multiple decision trees independently using bootstrapped sub-samples and random feature subsets. Tree predictions are averaged to reduce variance and prevent overfitting.
* **Config**: `n_estimators=150`, `max_depth=10`, `n_jobs=-1`


* **Extra Trees (`ExtraTreesClassifier`)**: Extremely Randomized Trees select cut-points for splits completely at random for each candidate feature rather than searching for the optimal threshold. This introduces additional randomness, further reducing variance and accelerating training.
* **Config**: `n_estimators=150`, `max_depth=10`, `n_jobs=-1`



### 2. Boosting Models (Sequential Ensembles)

* **XGBoost (`XGBClassifier`)**: Trains decision trees sequentially, where each new tree aims to correct errors made by preceding trees using second-order gradients and $L1$/$L2$ regularization penalties.
* **Config**: `n_estimators=150`, `learning_rate=0.03`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`


* **LightGBM (`LGBMClassifier`)**: A leaf-wise tree growth framework that expands nodes with the highest loss reduction rather than growing level-wise.
* **Config**: `n_estimators=150`, `learning_rate=0.03`, `max_depth=5`



### 3. Meta-Voting Strategies

* **Hard Voting (`voting='hard'`)**: Collects discrete class predictions from all four base estimators and outputs the majority class vote.
* **Soft Voting (`voting='soft'`)**: Averages the predicted class probabilities across all estimators and selects the class with the highest combined probability:

$$\hat{y} = \arg\max_k \sum_{i=1}^{M} P_i(y = k \mid X)$$

---

## ⚙️ Model Comparison Summary

| Model | Type | Tree Growth Strategy | Primary Role |
| --- | --- | --- | --- |
| **Random Forest** | Bagging | Level-wise | Controls variance, provides stable base predictions |
| **Extra Trees** | Bagging | Level-wise | Increases randomness to reduce overfitting and speed up splits |
| **XGBoost** | Boosting | Level-wise | Minimizes residual errors via regularized gradient boosting |
| **LightGBM** | Boosting | Leaf-wise | High-efficiency optimization on sparse, encoded features |
| **Voting Ensemble** | Meta | N/A | Combines bagging stability with boosting accuracy |

---

## 📈 Ensemble Learning Performance

The base models are combined using two voting strategies via `sklearn.ensemble.VotingClassifier`:

### 1. Hard Voting Ensemble (`voting='hard'`)

* Aggregates predictions across all 4 base models via majority vote.
* **Test Accuracy**: `~82.09%`

```text
=======================================================
     HARD VOTING ENSEMBLE RESULTS (Accuracy: 0.8209)     
=======================================================
              precision    recall  f1-score   support

     Average       0.64      0.91      0.75       395
        High       0.73      0.61      0.67       243
         Low       0.96      0.84      0.89       976

    accuracy                           0.82      1614
   macro avg       0.78      0.79      0.77      1614
weighted avg       0.85      0.82      0.83      1614

```

### 2. Soft Voting Ensemble (`voting='soft'`)

* Computes and averages the predicted class probabilities across all 4 base models to select the highest average probability class.
* **Test Accuracy**: `~81.85%`

```text
=======================================================
     SOFT VOTING ENSEMBLE RESULTS (Accuracy: 0.8185)     
=======================================================
              precision    recall  f1-score   support

     Average       0.64      0.89      0.75       395
        High       0.73      0.61      0.67       243
         Low       0.95      0.84      0.89       976

    accuracy                           0.82      1614
   macro avg       0.77      0.78      0.77      1614
weighted avg       0.84      0.82      0.82      1614

```

---

## 🚀 Getting Started

### Prerequisites

Python 3.8+ is required. Install necessary dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn datasets scikit-learn xgboost lightgbm jupyter

```

### Setup & Execution

1. Clone the repository:
```bash
git clone https://github.com/your-username/customer-spending-prediction.git
cd customer-spending-prediction

```


2. Launch Jupyter Notebooks:
```bash
jupyter notebook

```


3. Run the Notebooks in order:
* Open `EDA.ipynb` to review initial data visualizations and distributions.
* Open `models.ipynb` to inspect individual baseline performance and experiments.
* Open and run all cells in `main.ipynb` to execute the full data preprocessing, feature engineering, and ensemble training pipeline.



---

## 📊 Output & Visualization

Running `main.ipynb` generates performance logs and comparison plots:

* **Classification Reports**: Displays precision, recall, and F1-scores for both Hard and Soft Voting ensembles.
* **Confusion Matrix Plot**: Renders side-by-side Seaborn heatmaps (`cmap='Blues'` vs `cmap='Greens'`) comparing prediction accuracy and classification behavior across spending categories for both voting mechanisms.
