# Diabetes Prediction (Medium Testing)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DANIAL3304/diabetes_prediction_analysis/blob/main/diabetes_analysis_medium_testing.ipynb)

This repo has my part of our group's data mining project on diabetes prediction. The full project is on our website: https://sites.google.com/view/diabetes-gigabyte/home

We trained all the models on one large dataset and tested them on several other datasets. I did the **medium testing**: training on the BRFSS survey dataset (70,692 records) and testing on a separate clinical dataset (10,000 records).

## Background

Diabetes is often not detected until complications appear. Most diabetes datasets only get used for yes/no prediction, so in this project we also wanted to find patient groups and the risk factors behind the predictions. We used three techniques:

- Classification to predict whether a person has diabetes
- Clustering to group patients with similar profiles
- Association rule mining to find risk factor combinations linked to diabetes

We also used explainable AI (SHAP, a surrogate decision tree, and an association flow plot) so the results can be interpreted.

## Experiment setup

| Scenario | Train | Test |
|---|---|---|
| **Medium (this repo)** | BRFSS 2015 (70,692) | Clinical dataset (10,000) |
| Small | BRFSS 2015 | Pima Indians (768) |
| Upscaling | BRFSS 2015 | Upscaled data |
| Downscaling | BRFSS 2015 | Downscaled data |

The training data comes from survey answers and the test data comes from clinical measurements. This scenario checks whether the models still work on data from a different source.

## Datasets

**Dataset 1: Diabetes Health Indicators (BRFSS 2015).** Used for training.
- File: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`
- 70,692 rows, 22 columns, all numeric, no missing values
- Balanced 50/50 between non-diabetic and diabetic/prediabetic

**Dataset 2: Diabetes Prediction Dataset.** Used for testing.
- File: `diabetes_data.csv`
- 10,000 rows, 11 columns (age, gender, BMI, blood pressure, cholesterol, glucose, diabetes_probability, plus category columns), no missing values
- This is a simulated dataset

The CSV files are not included here. The download links are on the [Overview page](https://sites.google.com/view/diabetes-gigabyte/overview) of our site.

## Preprocessing

The two datasets have different columns, so I reduced both to the same five: `Diabetes_binary`, `HighBP`, `BMI`, `Sex` and age.

For the BRFSS dataset, age is stored as a category from 1 to 13, so I mapped each category to an approximate age (1 = 21, 2 = 27, ..., 13 = 82).

For the clinical dataset:
- `Diabetes_binary` is 1 when `diabetes_probability > 0.5`
- `HighBP` is 1 when systolic > 130 or diastolic > 80
- `gender` is converted to `Sex` (Male = 1, Female = 0)
- `bmi` and `age` are renamed to match the BRFSS dataset

After that, I cast the test set's column types to match the training set (all `float64`) so the models read both the same way.

## Feature engineering

I added three interaction features, because diabetes risk usually comes from combinations of factors rather than one factor alone:

| Feature | Formula | Idea |
|---|---|---|
| `Metabolic_Score` | BMI × (HighBP + 1) | Obesity and high blood pressure together |
| `Age_Weighted_BMI` | BMI × (age / 100) | High BMI matters more at an older age |
| `Age_HighBP` | age × HighBP | High blood pressure matters more at an older age |

Age is also normalised to 0–1 (`Age_Normalized`). The final input has 7 features: `HighBP`, `BMI`, `Sex`, `Age_Normalized`, `Metabolic_Score`, `Age_Weighted_BMI`, `Age_HighBP`.

## Methods

**Classification:** Decision Tree, Random Forest, XGBoost, Logistic Regression, kNN, SVM (RBF), Gaussian Naive Bayes, and ANN (MLP).

**Clustering:** k-Means, Agglomerative (hierarchical) clustering, and DBSCAN.

**Association rules:** Apriori and FP-Growth.

Other details:
- Hyperparameters were tuned with `RandomizedSearchCV` (5 iterations, 3-fold CV, F1 scoring). GridSearchCV was too slow on Colab.
- `StandardScaler` was fitted on the training set only.
- kNN, SVM and ANN were trained on a 15,000-row sample to save time.
- For clustering, k was chosen from 2 to 5 using the silhouette score.
- For association rules, I tried min support values of 0.2, 0.1 and 0.05, with lift ≥ 1.2.

## Results

### Classification

The test set has 5,266 non-diabetic and 4,734 diabetic records. Precision, recall and specificity were calculated from the confusion matrices.

| Model | Accuracy | F1 | Precision | Recall | Specificity |
|---|---|---|---|---|---|
| SVM | 0.7643 | 0.7469 | 0.760 | 0.735 | 0.791 |
| ANN (MLP) | 0.7621 | 0.7310 | 0.787 | 0.683 | 0.833 |
| XGBoost | 0.7587 | 0.7291 | 0.778 | 0.686 | 0.824 |
| Random Forest | 0.7539 | 0.7260 | 0.767 | 0.689 | 0.812 |
| Gaussian NB | 0.7559 | 0.7233 | 0.780 | 0.674 | 0.829 |
| Logistic Regression | 0.7611 | 0.7232 | 0.801 | 0.659 | 0.853 |
| Decision Tree | 0.7465 | 0.7189 | 0.757 | 0.685 | 0.802 |
| kNN | 0.7046 | 0.6764 | 0.703 | 0.652 | 0.752 |

SVM had the best accuracy and F1. It also had the highest recall: it missed 1,256 diabetic cases, the fewest of any model. Missing a diabetic patient is the worse mistake in screening, so this matters more than accuracy alone. Logistic Regression went the other way, with the fewest false positives (776) but more missed cases. kNN was the weakest, probably because the survey and clinical data are distributed differently.

Most models landed around 75–76% accuracy, which suggests the features still work on data from a different source.

<details>
<summary>Confusion matrices</summary>

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| SVM | 4165 | 1101 | 1256 | 3478 |
| ANN (MLP) | 4389 | 877 | 1502 | 3232 |
| XGBoost | 4340 | 926 | 1487 | 3247 |
| Random Forest | 4278 | 988 | 1473 | 3261 |
| Gaussian NB | 4368 | 898 | 1543 | 3191 |
| Logistic Regression | 4490 | 776 | 1613 | 3121 |
| Decision Tree | 4224 | 1042 | 1493 | 3241 |
| kNN | 3959 | 1307 | 1647 | 3087 |

</details>

### Clustering

| Algorithm | Silhouette score |
|---|---|
| k-Means (k = 2) | 0.4184 |
| Hierarchical (k = 2) | 0.4178 |
| DBSCAN | 0.3065 |

k-Means and hierarchical clustering gave almost the same result: two fairly clear groups. DBSCAN produced many small clusters and noise points, so it didn't suit this data.

### Association rules

| Algorithm | Min support | Rules | Avg confidence | Avg lift |
|---|---|---|---|---|
| Apriori | 0.05 | 2966 | 0.5016 | 1.8394 |
| FP-Growth | 0.05 | 2966 | 0.5016 | 1.8394 |

Both algorithms found the same rules. That's expected, because they only differ in how they search for frequent itemsets. An average lift of 1.84 means the antecedents and the consequent appear together about 1.84 times more often than they would by chance.

## Explainability

- **SHAP (classification):** the top three features were `Age_Weighted_BMI`, `Metabolic_Score` and `Age_HighBP`, which are the three features I engineered. The raw features (`BMI`, `HighBP`, `Sex`, `Age_Normalized`) mattered much less on their own.
- **Surrogate decision tree (clustering):** the first split is `Age_HighBP <= 0.368`. Age-related high blood pressure is the main thing separating the lower-risk group from the higher-risk group.
- **Association flow plot:** `Metabolic_Risk_High` had the strongest link to diabetes (lift around 9.7), and `Metabolic_Risk_Low` was linked to no diabetes. `Age_BP_Risk_High` had a stronger effect than `HighBP` alone.

## Limitations

- The clinical dataset is simulated, and its label comes from thresholding a probability, not from a real diagnosis.
- Only five features are shared between the two datasets, so glucose and cholesterol could not be used.
- Recall is between 0.65 and 0.74, so a fair number of diabetic cases are still missed. Adjusting the decision threshold could help if the model is used for screening.

## How to run

1. Open the notebook in Colab using the badge at the top.
2. Put both CSV files in `MyDrive/Datasets/Project/` in Google Drive, or change `DRIVE_PATH` in the first code cell.
3. Install any missing packages:
   ```
   !pip install xgboost mlxtend imbalanced-learn shap
   ```
4. Run all cells.

The notebook saves `experiment_results.json`, `scaler.pkl` and `diabetes_model.pkl` (the Random Forest) to the same Drive folder.

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, mlxtend, imbalanced-learn, shap, joblib

This project is for a university course and is not meant for medical use.
