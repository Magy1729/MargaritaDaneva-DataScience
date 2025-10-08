# House Price Prediction - Ames Housing Dataset
# Author: Margarita Biserova Daneva
# Goal: Predict house sale prices using exploratory data analysis (EDA), 
#       feature engineering, preprocessing, optimization; Models tested (Linear, Ridge
#       Lasso, ElasticNet, RandomForest, KFolds); hiperparameter tuning. 

# =====================
# 1. Libraries
# =====================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# =====================
# 2. Load dataset
# =====================

df = pd.read_csv(r"C:\Users\Usuario\Desktop\Exercise\My_house_pricing project\Data\AmesHousing.csv")

# Copy dataset to work on it
mydf = df.copy()

# =========================
# 3. First look at data
# =========================
print(mydf.shape)        # shape: (rows, columns)
print(mydf.info())       # column names and types
print(mydf.describe())   # numerical summary
print(mydf.isna().sum().sort_values(ascending=False)) # missing values

# Separate numeric and categorical features
numeric_data = mydf.select_dtypes(include=[np.number])
categorical_data = mydf.select_dtypes(exclude=[np.number])
print("\nNumeric columns:", numeric_data.columns)
print("\nCategorical columns:", categorical_data.columns)

# ==========================
# 4. Correlation with target
# ==========================
def correlacion_var_saleprice(df, target="SalePrice"):
    resultados = []
    for col in numeric_data.columns:
        coef, pval = pearsonr(df[col].fillna(0), df[target])
        resultados.append((col, coef, pval))
    results = pd.DataFrame(resultados, columns=["variable", "correlacion", "p_value"])
    return results.sort_values(by="correlacion", ascending=False, key=abs)

print(correlacion_var_saleprice(mydf))

# Heatmap of selected features
plt.figure(figsize=(6, 20))
corr_target = numeric_data.corr()[["SalePrice"]].sort_values(by="SalePrice", ascending=False)
sns.heatmap(corr_target, cmap="coolwarm", annot=True)
plt.title("Correlation of numerical features with SalePrice")
plt.show()

# ========================
# 5. Handle missing values
# ========================
# Example: Lot Frontage → fill by neighborhood median

medians = mydf.groupby("Neighborhood")["Lot Frontage"].median()

mydf["Lot Frontage"] = mydf.apply(
    lambda row: medians[row["Neighborhood"]] if pd.isna(row["Lot Frontage"]) else row["Lot Frontage"],
    axis = 1
)

medians_lot_frontage = mydf["Lot Frontage"].mean()
mydf["Lot Frontage"] = mydf["Lot Frontage"].fillna(medians_lot_frontage)


# Fill other categorical missing values with "None"
categorical_cols = mydf.select_dtypes(exclude=[np.number]).columns
mydf[categorical_cols] = mydf[categorical_cols].fillna("None")

# Fill specific numeric missing with 0
mydf["Mas Vnr Area"] = mydf["Mas Vnr Area"].fillna(0)
mydf["Garage Area"] = mydf["Garage Area"].fillna(0)
mydf["Bsmt Full Bath"] = mydf["Bsmt Full Bath"].fillna(0)
mydf["Bsmt Half Bath"] = mydf["Bsmt Half Bath"].fillna(0)

# =======================
# 6. Feature engineering
# =======================
# Age of the house at time of sale
mydf["House_Age"] = (mydf["Yr Sold"] - mydf["Year Built"]).clip(lower=0)

# Age since remodeling
mydf["Remod_Age"] = (mydf["Yr Sold"] - mydf["Year Remod/Add"]).clip(lower=0)

# Total bathrooms (full + half from basement and above ground)
mydf["Total_Bath"] = (
    mydf["Full Bath"] + 0.5 * mydf["Half Bath"] +
    mydf["Bsmt Full Bath"] + 0.5 * mydf["Bsmt Half Bath"]
)

# Total porch area
mydf["Total_Porch"] = (
    mydf["Open Porch SF"] + mydf["Enclosed Porch"] + mydf["3Ssn Porch"]
)

# Binary indicators
mydf["Has Garage"] = (mydf["Garage Area"] > 0).astype(int)
mydf["Has Pool"] = (mydf["Pool Area"] > 0).astype(int)

# ==========================
# 7. Rare category grouping
# ==========================
skip_group = [
    "Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond","Bsmt Exposure",
    "BsmtFin Type 1","BsmtFin Type 2","Heating QC","Kitchen Qual",
    "Fireplace Qu","Garage Qual","Garage Cond","Pool QC","Paved Drive"
]
cols_to_group = [c for c in categorical_cols if c not in skip_group]

threshold = 10
for col in cols_to_group:
    vc = mydf[col].value_counts(dropna=False)
    rare = vc[vc < threshold].index
    if len(rare) > 0:
        mydf[col] = mydf[col].replace(list(rare), "Other")

# =========================================
# 8. Log transformation for skewed features
# =========================================
# Helps normalize distributions and improve model performance

def log_transformation(df, cols):
    """Apply log(1+x) transformation to reduce skewness."""
    for col in cols:
        df[f"Log_{col}"] = np.log1p(df[col])
    return df

# Columns identified as skewed during EDA
columns_for_log = [
    "SalePrice", "Lot Area", "Gr Liv Area",
    "1st Flr SF", "2nd Flr SF", "Total Bsmt SF",
    "Mas Vnr Area", "BsmtFin SF 1", "Bsmt Unf SF",
    "Wood Deck SF", "Open Porch SF", "Enclosed Porch",
    "3Ssn Porch", "Screen Porch"
]

mydf = log_transformation(mydf, columns_for_log)

# Quick check: visualize one example before vs after
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
mydf["Gr Liv Area"].hist(bins=30)
plt.title("Original Gr Liv Area")

plt.subplot(1,2,2)
mydf["Log_Gr Liv Area"].hist(bins=30)
plt.title("Log-Transformed Gr Liv Area")

plt.show()

# =====================
# 9. Orginal encoding
# =====================

qual_cond_map = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}
bsmt_exposure_map = {"None":0, "No":1, "Mn":2, "Av":3, "Gd":4}
bsmt_fin_map = {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6}
paved_drive_map = {"N":0, "P":1, "Y":2}

# Apply quality/condition mapping
for col in ["Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond",
            "Heating QC","Kitchen Qual","Fireplace Qu",
            "Garage Qual","Garage Cond","Pool QC"]:
    mydf[col] = mydf[col].map(qual_cond_map)

# Apply basement exposure mapping
mydf["Bsmt Exposure"] = mydf["Bsmt Exposure"].map(bsmt_exposure_map)

# Apply basement finish type mapping
for col in ["BsmtFin Type 1", "BsmtFin Type 2"]:
    mydf[col] = mydf[col].map(bsmt_fin_map)

# Apply paved drive mapping
mydf["Paved Drive"] = mydf["Paved Drive"].map(paved_drive_map)

# =====================
# 10. Next steps 
# =====================

# Handle missing values in basement and garage features
print(mydf["Log_Total Bsmt SF"].isna().sum())


# Bsmt Exposure (NaN means "no basement") → fill with 0
mydf["Bsmt Exposure"] = mydf["Bsmt Exposure"].fillna(0)

# Fill NaN in numeric basement/garage features with 0
columns_with_nan = [
    "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Garage Yr Blt", "Garage Cars",
    "Log_Total Bsmt SF", "Log_BsmtFin SF 1", "Log_Bsmt Unf SF"
    ]

for col in columns_with_nan:
    mydf[col] = mydf[col].fillna(0)

# Confirm that no missing values remain
print("Total NaN values remaining:", mydf.isna().sum().sum())


# - One-hot encoding for nominal categorical features
nominal_features = mydf.select_dtypes(exclude=[np.number]).columns
mydf = pd.get_dummies(mydf, columns=nominal_features, drop_first=True, dtype=int)
# - Train/test split

# - Baseline modeling (Linear Regression)
y = mydf["Log_SalePrice"]
X = mydf.drop(columns=["Log_SalePrice", "SalePrice"])

lm = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm.fit(X, y)
y_lm_pred = lm.predict(X_test)
y_pred = lm.predict(X_test)

rmse_lm = np.sqrt(mean_squared_error(y_test, y_pred))
mae_lm = mean_absolute_error(y_test, y_pred)
r2_lm = r2_score(y_test, y_pred)


#KDE plot for observed vs Predicted Values
plt.figure(figsize=(8,5))

sns.kdeplot(y_test, label="Actual", fill=True, color="blue")
sns.kdeplot(y_pred, label="Predicted", fill=True, color="red")

plt.title("KDE plot of Actual vs Predicted Price")
plt.xlabel("Actual Price (blue)")
plt.ylabel("Predicted Price (red)")
plt.legend()

plt.show()

#=============================
# Pipelines
#=============================

# RidgeCV Pipeline
alphas = [0.01, 0.1, 1, 10, 100]
ridge_pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, store_cv_results=True))

ridge_pipe.fit(X_train, y_train)
y_ridge_pred = ridge_pipe.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_ridge_pred))
r2_ridge = r2_score(y_test, y_ridge_pred)

# Lasso Pipeline
lasso_pipe = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5))

lasso_pipe.fit(X_train, y_train)
y_lasso_pred = lasso_pipe.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_lasso_pred))
r2_lasso = r2_score(y_test, y_lasso_pred)

# RandomForest Pipeline
random_forest_pipe = make_pipeline(RandomForestRegressor(n_estimators=100, max_depth=5))
random_forest_pipe.fit(X_train, y_train)
y_random_pred= random_forest_pipe.predict(X_test)

mse_random = mean_squared_error(y_test, y_random_pred)
rmse_random = np.sqrt(mean_squared_error(y_test, y_random_pred))
r2_random = r2_score(y_test, y_random_pred)


# ElsaticNet Pipeline
elastic_net_pipe = make_pipeline(StandardScaler(), ElasticNetCV(alphas=alphas, cv=5, l1_ratio=[0.1, 0.5, 0.9]))
elastic_net_pipe.fit(X_train, y_train)
y_elastic_pred = elastic_net_pipe.predict(X_test)

mse_elastic = mean_squared_error(y_test, y_elastic_pred)
rmse_elastic = np.sqrt(mean_squared_error(y_test, y_elastic_pred))
r2_elastic = r2_score(y_test, y_elastic_pred)



# KFolds
model = LinearRegression()
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores_kfolds = cross_val_score(model, X, y, cv=kfolds, scoring="r2")
neg_mse_kfolds = cross_val_score(model, X, y, cv=kfolds, scoring="neg_mean_squared_error")
rmse_kfolds = np.sqrt(-neg_mse_kfolds)

print("R2 per fold: ", r2_scores_kfolds)           # model explain between 87% and 93% of the variance depending the fold
print("MSE per fold:", neg_mse_kfolds) 
print("R2 mean:", r2_scores_kfolds.mean())         


# ===============================
# Compare the Models
#================================
# Create list of results
results = [
    {"Model": "Ridge",
     "RMSE": rmse_ridge,
     "MSE": mse_ridge,
     "R^2": r2_ridge},

     {"Model": "Lasso",
      "RMSE": rmse_lasso,
      "MSE": mse_lasso,
      "R^2": r2_lasso},

      {"Model": "Random Forest",
       "RMSE": rmse_random,
       "MSE": mse_random,
       "R^2": r2_random},

       {"Model": "ElasticNet",
        "RMSE": rmse_elastic,
        "MSE": mse_elastic,
        "R^2": r2_elastic}
]

df_results = pd.DataFrame(results)

df_results = df_results.sort_values(by="R^2", ascending=False)
print("\n=======Model Compare========")
print(df_results)

# - Hyperparameter tuning
#=======================
# Feature importances
#=====================
rf_model = random_forest_pipe.named_steps["randomforestregressor"]
importances = rf_model.feature_importances_

features = X_train.columns
feat_importances = pd.Series(importances, index=features)

feat_importances.nlargest(10).plot(kind="barh", figsize=(10,6))
plt.gca().invert_yaxis()
plt.title("Top 10 feature importances Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

