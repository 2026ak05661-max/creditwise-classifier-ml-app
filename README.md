# Credit Risk Classification – 5 ML Models with Streamlit

## a. Problem statement

A bank wants to decide whether a loan applicant is a **good credit risk (0)** or a **bad credit risk (1)**, using the applicant's financial and personal details. This is a **binary classification** problem.

Approving a bad-risk applicant is costly for the bank, so finding the bad-risk class (recall for class 1) matters as much as overall accuracy.

Five classification models are trained on the same dataset and compared on six evaluation metrics. They are then served through an interactive Streamlit web app deployed on Streamlit Community Cloud.

## b. Dataset description

| Item | Details |
|---|---|
| Name | Statlog (German Credit Data) |
| Source | UCI Machine Learning Repository (Prof. Hans Hofmann, Universität Hamburg); also on Kaggle |
| Instances | 1,000 loan applicants (minimum required: 500) |
| Features | 20 (minimum required: 12): 7 numeric and 13 categorical |
| Target | `credit_risk`: 0 = Good (700), 1 = Bad (300), so the classes are imbalanced at 70 : 30 |
| Missing values | None |

| Type | Features |
|---|---|
| Numeric (7) | duration_months, credit_amount, installment_rate, residence_since, age, existing_credits, num_dependents |
| Categorical (13) | checking_status, credit_history, purpose, savings, employment_since, personal_status_sex, other_debtors, property, other_installment_plans, housing, job, own_telephone, foreign_worker |

**Pre-processing:**

- The original UCI category codes (A11, A12, …) are converted to readable labels using the dataset's code book.
- Numeric features are scaled with `StandardScaler`.
- Categorical features are one-hot encoded with `OneHotEncoder`.
- Both steps sit inside each model's saved `Pipeline`, so the app can use the raw CSV directly.

**Train/test split:** 80% / 20%, stratified by class (`random_state = 42`). That gives 800 training rows and 200 test rows. The test rows are saved as `test_data.csv`, with 140 good and 60 bad.

**Class imbalance:** `class_weight = "balanced"` is used for Logistic Regression, Decision Tree and SVM.

## c. GitHub repository link

**Repository:** [https://github.com/<your-username>/<your-repo-name>](https://github.com/2026ak05661-max/creditwise-classifier-ml-app)

**Live Streamlit app:** [https://<your-app-name>.streamlit.app](https://2026ak05661-max-creditwise-classifier-ml-app-app-9i2wco.streamlit.app/)

```
project-folder/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # test data used in the experiments (features + target)
│-- data/
│   │-- german_raw.csv         # original UCI file (coded values)
│   └── german_credit.csv      # decoded full dataset
│-- model/
    │-- train_models.py        # source code: pre-processing, training, evaluation
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- support_vector_machine.pkl
    │-- feature_names.pkl      # preprocessing artifact (feature order)
    │-- metrics.csv            # comparison table
    └── sklearn_version.txt
```

## d. Models used

The five models are:

1. Logistic Regression (`class_weight = "balanced"`, `max_iter = 2000`)
2. Decision Tree Classifier (`max_depth = 3`, `min_samples_leaf = 10`, `class_weight = "balanced"`)
3. K-Nearest Neighbour Classifier (`k = 7`)
4. Naive Bayes (Gaussian)
5. Support Vector Machine (RBF kernel, `class_weight = "balanced"`)

### Comparison table (test data, n = 200)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7500 | 0.8058 | 0.5581 | 0.8000 | 0.6575 | 0.4893 |
| Decision Tree | 0.6300 | 0.6959 | 0.4397 | 0.8500 | 0.5795 | 0.3581 |
| KNN | 0.7350 | 0.7357 | 0.6000 | 0.3500 | 0.4421 | 0.3015 |
| Naive Bayes | 0.6950 | 0.7411 | 0.4937 | 0.6500 | 0.5612 | 0.3415 |
| Support Vector Machine | 0.7600 | 0.8032 | 0.5750 | 0.7667 | 0.6571 | 0.4900 |

*Precision, Recall and F1 are calculated for the positive class (1 = Bad credit risk). AUC uses predicted probabilities, or the SVM's decision-function scores.*

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best balance. It has the highest AUC (0.8058), F1 (0.6575) and the second-highest MCC (0.4893). It catches 48 of the 60 bad-risk applicants (recall 0.80) with 38 false alarms. The balanced class weights stop it from favouring the majority "good" class. It is also simple and its coefficients are easy to explain, which matters in banking. |
| Decision Tree | Lowest accuracy (63%) and AUC (0.6959). Being shallow (depth 3) and balanced, it flags many applicants as bad. That gives the highest recall (0.85, only 9 bad applicants missed) but the lowest precision (0.44, 65 false alarms). Deeper trees overfitted and did worse, so its simple axis-parallel rules cannot capture this data well. |
| KNN | Accuracy (73.5%) looks reasonable, but this is misleading. Recall is only 0.35: it misses 39 of the 60 bad-risk applicants, which gives the lowest F1 (0.4421) and MCC (0.3015). With no class weighting, the 70% majority "good" class dominates each neighbourhood, and distances over many one-hot features carry little information. |
| Naive Bayes | Middle performance: accuracy 69.5%, AUC 0.7411, recall 0.65, MCC 0.3415. It assumes all features are independent and that each one follows a Gaussian distribution. Neither holds for one-hot categorical features or related features such as duration and credit amount, so its results are limited. |
| Support Vector Machine | Highest accuracy (76%) and MCC (0.4900), with an AUC of 0.8032, essentially tied with Logistic Regression. It catches 46 of the 60 bad-risk applicants (recall 0.7667) with 34 false alarms. The RBF kernel captures some non-linear patterns, but the gain over the linear model is very small. |
| **Overall winner for this dataset** | **Logistic Regression**, with SVM a very close second. SVM is slightly ahead on accuracy and MCC, while Logistic Regression has the best AUC, F1 and recall and misses fewer bad-risk applicants (12 vs 14). The UCI documentation rates missing a bad applicant as 5 times more costly than rejecting a good one, so that matters most here. Logistic Regression is also the more interpretable model for credit decisions. Accuracy alone is not a reliable guide on this imbalanced dataset; KNN shows why. |

## Streamlit app features

The app (`app.py`) includes:

1. **Dataset upload (CSV):** upload test data containing all 20 features and the `credit_risk` column. A sample `test_data.csv` can be downloaded from the sidebar.
2. **Model selection dropdown:** choose any of the 5 saved models.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC.
4. **Confusion matrix and classification report** for the selected model.
5. **Comparison table** of all 5 models on the uploaded test data.

The app only **loads** the saved `.pkl` models from `model/`. It does not retrain on the uploaded data.

## How to run

```bash
pip install -r requirements.txt
python model/train_models.py   # trains the models, saves the .pkl files, test_data.csv and metrics.csv
streamlit run app.py
```
