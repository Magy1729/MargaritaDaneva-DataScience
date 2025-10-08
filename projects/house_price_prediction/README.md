House Price Prediction – Ames Housing Dataset

Introduction
The goal of this project is to build a predictive model to estimate house sale prices using the Ames Housing dataset (≈2930 observations, 82 features).
The project aims to:
	- Explore and clean the dataset.
	- Perform exploratory data analysis (EDA) to identify trends and correlations.
	- Apply feature engineering to create meaningful new variables.
	- Train and evaluate multiple machine learning models for prediction.

Exploratory Data Analysis (EDA)
Data size: 2930 rows × 82 features (numerical + categorical).
Key visualizations used: histograms, boxplots, scatter plots, regression plots, heatmaps.
Findings:
SalePrice and LotArea were right-skewed → log transformation applied.
Strongest correlations with SalePrice: OverallQual (0.80), GrLivArea (0.71), 1stFlrSF (0.62).
Outliers (e.g., very large lots, very large houses) were real cases, not errors.

Data Cleaning & Feature Engineering
Missing values handled via:
Neighborhood medians for LotFrontage.
"None" for categorical NAs (e.g., PoolQC, Alley, Fence).
0 for numeric NAs (e.g., basement/garage areas).
New engineered features:
House_Age = YrSold – YearBuilt.
Remod_Age = YrSold – YearRemodAdd.
Total_Bath (full, half, basement baths).
HasGarage, HasPool (binary).
Encoding: one-hot for nominal categories; ordinal mapping for quality ratings.
Applied log(1+x) transformation to reduce skewness in key features and SalePrice.

Modeling & Evaluation
Models trained:
Linear Regression (baseline) → R² ≈ 0.95, low RMSE, strong baseline.
Ridge Regression (CV) → Best α ≈ 1.0. Stable performance, no overfitting.
Lasso Regression (CV) → Best α ≈ 0.01. Slightly lower accuracy, performed feature selection.
ElasticNet (CV) → Showed best trade-off (L1+L2 regularization). Coefficient paths analyzed.
Random Forest Regressor → Robust but slightly worse generalization (R² ~0.87).

Model Comparison
Model		RMSE	MSE	R²
ElasticNet	0.1124	0.0126	0.9317
Ridge		0.1131	0.0128	0.9309
Lasso		0.1204	0.0145	0.9216
RandomForest	0.1517	0.0230	0.8757

Interpretation:
ElasticNet achieved the highest R² and lowest error, making it the best-performing model.
Ridge was nearly identical, showing regularization works well here.
Lasso selected fewer features but had slightly higher error.
Random Forest underperformed compared to linear models.

Conclusion
The final selected model is ElasticNet Regression, which provided:
The best predictive accuracy.
Strong generalization ability.
Automatic feature selection through L1 regularization.
This project demonstrates the full ML workflow: EDA → Cleaning → Feature Engineering → Modeling → Optimization → Model Selection.
