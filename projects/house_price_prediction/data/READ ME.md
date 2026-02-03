House Price Prediction – Ames Housing

Overview
This project builds regression models to predict house sale prices using the Ames Housing dataset (~2,900 observations, 82 features). The focus is on feature engineering, handling skewed data, and comparing linear and tree-based models.

Dataset
Ames Housing dataset (Kaggle).
Target variable: SalePrice (log-transformed).

Key Steps
Exploratory Analysis
Strong correlations with OverallQual, GrLivArea, 1stFlrSF, and YearBuilt.
Several numerical features showed strong right-skewness (SalePrice, LotArea, GrLivArea).

Data Cleaning & Feature Engineering
Missing values interpreted semantically (e.g. no garage / no basement → 0 or “None”).
Log(1+x) transformation applied to skewed numerical variables.

Engineered features:
house_age = YrSold - YearBuilt
remod_age = YrSold - YearRemodAdd
Binary indicators for sparse continuous features (e.g. HasGarage, HasPool).
One-hot encoding for categorical variables.

Modeling
Models evaluated:
Linear Regression (baseline)
Ridge Regression (CV)
Lasso Regression (CV)
Random Forest Regressor

Why Ridge?
Linear models performed very well after feature engineering, but multicollinearity was present. Ridge provided better stability than standard Linear Regression while preserving predictive power.
Results (test set, log-scale)
Ridge Regression: R² ≈ 0.95, RMSE ≈ 0.09, MAE ≈ 0.07
Random Forest Regressor: R² ≈ 0.87 (lower interpretability, higher complexity)
Despite strong Random Forest performance, Ridge was preferred for interpretability and stability.

Validation
K-Fold Cross-Validation used to assess generalization.
Similar train/test scores → no strong overfitting detected.
Limitations & Next Steps
Potential information leakage due to strong engineered features.
Feature importance could be extended using SHAP or permutation importance.
Test performance on a truly unseen dataset.

This project demonstrates the full ML workflow: EDA → Cleaning → Feature Engineering → Modeling → Optimization → Model Selection.


