import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV
from sklearn.pipeline import  make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv(r"C:\Users\Usuario\Desktop\Exercise\My_house_pricing project\Data\AmesHousing.csv")

# Inderstand the Dataset

print(df.shape)                 # para saber cuantas filas y columnas contiene
print(df.info())                # para identificar el nombre de las columnas 
print(df.dtypes)                # para saber tipo de los datos en cada columna
print(df.describe())            # resumen de las columnas numericas
print(df.describe().count())    # numero de valores no nulos


# Make a copy of the Dataset and work on it
mydf = df.copy()

#=================================================================
# EDA
#=================================================================

# Ver columnas con NaN y contar calores faltantes
missing_data = mydf.isna().sum().sort_values(ascending=False)
print(missing_data[missing_data > 0])


# Separate numerical and categories data
numeric_data = mydf.select_dtypes(include=[np.number])
categorical_data = mydf.select_dtypes(exclude=[np.number])
print("\nNumerical datas are: ", numeric_data.columns)
print("\nCategorical datas are: ", categorical_data.columns)

# Scater plot function of all numeric columns
dropped_price = numeric_data.drop(columns=["SalePrice"]).columns


#def scatter_numeric_vs_price(df, dropped_price, target="SalePrice"):
    #for col in dropped_price:
        #plt.figure(figsize=(6,4))
        #plt.scatter(df[col], df[target], alpha=0.5)
        #plt.xlabel(col)
        #plt.ylabel(target)
        #plt.title(f"{col} vs {target}")
        #plt.show()
        #plt.close()

#scatter_numeric_vs_price(mydf, dropped_price)


# Histogramas
#for col in numeric_data.columns:
    #plt.figure(figsize=(8,6))
    #sns.histplot(mydf[col].dropna(), kde=True)
    #plt.title(f"Distribución de {col}")
    #plt.show()
    #plt.close()

# Boxplots
#for col in numeric_data.columns:
    #plt.figure(figsize=(8,6))
    #sns.boxplot(x=mydf[col].dropna())
    #plt.title(f"Boxplot de {col}")
    #plt.show()
    #plt.close()

# Reg plot
#for col in numeric_data.columns:
    
    #sns.regplot(x=col, y="SalePrice", data=mydf)
    #plt.title(f"Regression plot of a{col} columns vs Sale Price")
    #plt.xlabel({col})
    #plt.ylabel("Sale Price")
    #plt.ylim(0,)
    #plt.show()


# Value counts de categóricas
for col in categorical_data.columns:
    print(f"\n{col}:\n", mydf[col].value_counts())

# Correlation with SalePrice
from scipy.stats import pearsonr

def correlacion_var_saleprice(mydf, target="SalePrice"):
    resultados = []
    for col in numeric_data.columns:
        coef, pval = pearsonr(mydf[col], mydf[target])
        resultados.append((col, coef, pval))
    results =  pd.DataFrame(resultados,  columns=["variable", "correlacion", "p_value"]) 
    ranking = results.sort_values(by="correlacion", ascending=False, key=abs)
    return ranking

print(correlacion_var_saleprice(mydf))

#=======================================================================
# Heat map for LotArea and GrLivArea
#=======================================================================

# Selection relation columns
df_heatmap = mydf[["Gr Liv Area", "Lot Area", "SalePrice"]]


# Grafic the Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(df_heatmap.corr(), cmap="coolwarm",
            annot=True,)

plt.title("Precio promedio segun Gr Living Area and Lot Area")
plt.xlabel("Lot Area")
plt.ylabel("Gr Living Area")
plt.show()


# Heatmap with numeric columns with Sale price
corr_target = numeric_data.corr()[["SalePrice"]].sort_values(by="SalePrice",ascending=False)

plt.figure(figsize=(6, 20))
sns.heatmap(corr_target, cmap="coolwarm", annot=True)
plt.title(" Precio promedio segun columnas numericas and Sale Price")
plt.xlabel(corr_target.columns)
plt.ylabel(corr_target.columns)
plt.show()

#=======================================================================
# Feature Engineering
#=======================================================================

# Values NaN
nan_values = np.isnan(numeric_data).sum()
print(nan_values)
avarage_Lot_frontage = mydf["Lot Frontage"].mean()
print(avarage_Lot_frontage)
print("\nColumns Lot Frontage", mydf["Lot Frontage"])   

# Seen the houses in same Neighborhood
close_neighborhood = mydf['Neighborhood'].value_counts()
print(close_neighborhood)

# replace NaN values in the Lot Frontage with the median of Neighborhood
medians = mydf.groupby("Neighborhood")["Lot Frontage"].median()

mydf["Lot Frontage"] = mydf.apply(
    lambda row: medians[row["Neighborhood"]] if pd.isna(row["Lot Frontage"]) else row["Lot Frontage"],
    axis = 1
)

# Confirm that column Lot Frontage has not NaN values
print("\nMissing values in Lot Frontage ", mydf["Lot Frontage"].isna().sum())

median_lot_frontage = mydf["Lot Frontage"].median()
mydf["Lot Frontage"] = mydf["Lot Frontage"].fillna(median_lot_frontage)
print(mydf["Lot Frontage"].isna().sum())


# Looking for NaN in the other columns in mydf
print(mydf.isna().sum().sort_values(ascending=False))

# What kind of values have Pool QC
print("\nPool QC is\n:", mydf["Pool QC"].value_counts())

# How many are NaN in Pool QC
print(mydf["Pool QC"].isna().value_counts())
mydf["Pool QC"] = mydf["Pool QC"].fillna("None")
print("\nCinfirm that all NaN are replaced with None", mydf["Pool QC"].isna().sum())
print(mydf['Pool Area'].value_counts())

# How many Nan has¡in Alley 
print(mydf[["Alley", "Street"]].isna().sum())
print(mydf[["Alley", 'Street', 'MS Zoning']])
mydf["Alley"] = mydf["Alley"].fillna("None")  # Replace it with None
print("\nAveriguar que Alley no tienen NaN ", mydf["Alley"].isna().sum())


# How many NaN has in Fence and replase it with None
print("\nSee how many NaN has Fence ", mydf["Fence"].isna().sum())
mydf["Fence"] = mydf["Fence"].fillna("None")
print("\nCheck that there is not Nan in Fence ", mydf["Fence"].isna().sum())


# How many NaN has Misc Feature
print("\nSee how many NaN has in Misc Feature ", mydf['Misc Feature'].isna().sum())
mydf["Misc Feature"] = mydf["Misc Feature"].fillna("None")
print("\nCheck that all Nan are replaced with None in Misc Feature", mydf["Misc Feature"].isna().sum())

# How many NaN hs Mas Vnr Type
print(mydf["Mas Vnr Type"].isna().sum())
mydf["Mas Vnr Type"] = mydf["Mas Vnr Type"].fillna("None")
print("\nCheck that all NaN are replaced with None in Mas Vnr Area", mydf["Mas Vnr Type"].isna().sum())


print(mydf["Mas Vnr Area"].isna().sum())
mydf["Mas Vnr Area"] = mydf["Mas Vnr Area"].fillna(0)
print("\nChack that all NaN are replaced with 0 in MasVnr Area ", mydf["Mas Vnr Area"].isna().sum())

#=======================================================================
# Additional engineering features:
#=======================================================================

# House_Age = Year Sold - Year Build
print(mydf["Yr Sold"].describe())
print(mydf["Year Built"].describe())

mydf["House_Age"] = mydf["Yr Sold"] - mydf["Year Built"]
print(mydf["House_Age"].describe())

# There was inconsitansy: the min age was -1 
# Correct if the house_age is less than 0, return 0.
mydf.loc[mydf["House_Age"] < 0, "House_Age"] = 0
print(mydf["House_Age"].describe())

# Remod_Age = Year Sold - Year Remod/Add
mydf["Remod_Age"] = mydf["Yr Sold"] - mydf["Year Remod/Add"]
print(mydf["Remod_Age"].describe())
mydf.loc[mydf["Remod_Age"] < 0, "Remod_Age"] = 0
print(mydf["Remod_Age"].describe())


#sns.regplot(x=mydf["House_Age"], y= mydf["SalePrice"], data=mydf)
#plt.ylim(0,)
#plt.title("Prueba de regresion plot sin log transformation")
#plt.xlabel("House age")
#plt.ylabel("Sale Price")
#plt.show()

# Total_Bath = Full Bath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath

mydf["Total_Bath"] = mydf["Full Bath"] + 0.5*mydf["Half Bath"] + mydf["Bsmt Full Bath"] + 0.5*mydf["Bsmt Half Bath"]
print(mydf["Total_Bath"].describe())       # count is 2928 and not 2930

 #Replace NaN in columns Bsmt Full Bath and Bsmt Half Bath with 0
mydf["Bsmt Full Bath"]=mydf["Bsmt Full Bath"].fillna(0)
mydf["Bsmt Half Bath"]=mydf["Bsmt Half Bath"].fillna(0)

# Re-check that all NaN are replaced
print("\nThe NaN values in column Bsmt Full Bath are: ", mydf["Bsmt Full Bath"].isna().sum())  
print("\nThe NaN values in column Bsmt Half Bath are: ", mydf["Bsmt Half Bath"].isna().sum())

mydf["Total_Bath"] = mydf["Full Bath"] + 0.5*mydf["Half Bath"] + mydf["Bsmt Full Bath"] + 0.5*mydf["Bsmt Half Bath"]


# Garage Area
mydf["Garage Area"] = mydf["Garage Area"].fillna(0)

# Total_Porch = Open Porch SF + Enclosed Porch + 3Ssn Porch
mydf["Total_Porch"] = mydf["Open Porch SF"] + mydf["Enclosed Porch"] + mydf["3Ssn Porch"]

# Create categorical values for Garage and Pool
mydf["Has Garage"] = (mydf["Garage Area"] > 0).astype(int)
mydf["Has Pool"] = (mydf["Pool Area"] > 0).astype(int)

# Functon grouping rare-category values on categorical data for reduce dimentionality
columns_categ = mydf.select_dtypes(exclude=[np.number]).columns
mydf[columns_categ] = mydf[columns_categ].fillna("None")

skip_group = [
    "Exter Qual","Exter Cond","Bsmt Qual","Bsmt Cond","Bsmt Exposure",
    "BsmtFin Type 1","BsmtFin Type 2","Heating QC","Kitchen Qual",
    "Fireplace Qu","Garage Qual","Garage Cond","Pool QC","Paved Drive"
]
cols_to_group = [c for c in columns_categ if c not in skip_group]


threshold = 10
for col in cols_to_group:
    vc = mydf[col].value_counts(dropna=False)
    rare = vc[vc < threshold].index
    if len(rare) > 0:
        mydf[col] = mydf[col].replace(list(rare), "Other")


print({c: mydf[c].nunique() for c in cols_to_group[:10]})

print(mydf[cols_to_group[2]].value_counts())

#===============================================
# Transformation and scaling for numerical data
#===============================================

from scipy import stats

# Log transformation for those columns that are right-skewed (see distribution plot)

def log_transformation(mydf, cols):
    for col in cols:
        mydf[f"Log {col}"] = np.log1p(mydf[col])
    return mydf

columns_for_log = ["Overall Qual", "Gr Liv Area", "SalePrice", "1st Flr SF", "2nd Flr SF", "Total Bsmt SF", "Mas Vnr Area", "BsmtFin SF 1", "Bsmt Unf SF", 
                       "Wood Deck SF", "Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch"]

mydf = log_transformation(mydf, columns_for_log)
print(mydf[[f"Log {col}" for col in columns_for_log]].head())


# Check if the transformation work (before and after):


#def plot_original_vs_log(mydf, cols):
    #for col in cols:
        #plt.figure(figsize=(8,6))
        #plt.subplot(1, 2, 1)
        #plt.subplot(1,2,1)
        #mydf[col].hist(bins=30)
        #plt.title(f"{col} original")

        #plt.subplot(1,2,2)
        #mydf[f"Log {col}"].hist(bins=30)
        #plt.title(f"{col} Log Transformation")   

        #plt.show()

#plot_original_vs_log(mydf, columns_for_log)

#===============================================
# Original-encoding using mapping dictionaries (e.g. Po=1, Fa=2, TA=3, Gd=4, Ex=5)
#===============================================

group_names_exter_qual = {"None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5}

columns_to_map = ["Exter Cond", "Bsmt Qual", "Bsmt Cond", "Heating QC", "Kitchen Qual", "Fireplace Qu"]

for col in columns_to_map:
    mydf[col] = mydf[col].map(group_names_exter_qual)
print(mydf[columns_to_map].value_counts())

col_to_map = ['Bsmt Exposure',"BsmtFin Type 1","BsmtFin Type 2", "Paved Drive"]


print(mydf["Bsmt Exposure"].unique())
print(mydf["BsmtFin Type 1"].unique())   
print(mydf["BsmtFin Type 2"].unique())
print(mydf["Paved Drive"].unique())

bsmt_fin_map = {"None":0, "Unf":1, "LwQ":2, "Rec":3, "BLQ":4, "ALQ":5, "GLQ":6}
bsmt_col_to_map = ["BsmtFin Type 1", "BsmtFin Type 2"]
for col in bsmt_col_to_map:
    mydf[col] = mydf[col].map(bsmt_fin_map)

print(mydf[bsmt_col_to_map].value_counts())

bsmt_explosure_to_map = {"NA":0, "No":1, "Mn":2, "Av":3, "Gd":4}
mydf["Bsmt Exposure"] = mydf["Bsmt Exposure"].map(bsmt_explosure_to_map)
print(mydf["Bsmt Exposure"].value_counts())

paved_drive_map = {"N":0, "P":1, "Y":2}
mydf["Paved Drive"] = mydf["Paved Drive"].map(paved_drive_map)
print(mydf["Paved Drive"].value_counts())

nominal_features = mydf.select_dtypes(exclude=[np.number]).columns

mydf=pd.get_dummies(mydf, columns=nominal_features, drop_first=True, dtype=int)

# Check dummies for neighborhood
print([c for c in mydf.columns if c.startswith("Neighborhood_")])

print("Neighborhood unique values before dummies:", df["Neighborhood"].nunique())
print("Neighborhood unique values after grouping:", mydf.filter(like="Neighborhood_").shape[1] + 1)
print(mydf.dtypes)


# Check is there unacceptable values
print("Any NaN?", mydf.isna().sum().sum())     # total NaN count
print("Any inf?", np.isinf(mydf).sum().sum())  # total inf count
print("Dtypes:\n", mydf.dtypes.value_counts()) # are all numeric?

# Deal with 250 NaN values
print(mydf.isna().sum()[mydf.isna().sum() > 0])
print(mydf["Bsmt Exposure"].value_counts())

#print(mydf[["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF", "Garage Yr Blt", "Garage Cars",
            #"Log Total Bsmt SF", "Log BsmtFin SF 1", "Log Bsmt Unf SF"]].value_counts())

mydf["Bsmt Exposure"] = mydf["Bsmt Exposure"].fillna(0)
print(mydf["Bsmt Exposure"].value_counts())

columns_with_nan = ["BsmtFin SF 1", "BsmtFin SF 2", "Total Bsmt SF", "Bsmt Unf SF", "Garage Yr Blt", "Garage Cars", "Log Total Bsmt SF", "Log BsmtFin SF 1", "Log Bsmt Unf SF"]
for col in columns_with_nan:
    mydf[col] = mydf[col].fillna(0)
    

print(f"Any NaN left? ", mydf.isna().sum().sum())

#===========================
# Linear Regression
#==========================

y = mydf["Log SalePrice"]
X = mydf.drop(columns=["Log SalePrice", "SalePrice"])

lm = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm.fit(X_train, y_train)

# Evaluate te model
y_pred = lm.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nThe mean squared error is : \n{rmse}")
print("\nThe R`2 score is :\n", r2)
print(f"\nThe absolute mean error is \n{mae}")

#KDE plot for observed vs Predicted Values
plt.figure(figsize=(8,5))

sns.kdeplot(y_test, label="Actual", fill=True, color="blue")
sns.kdeplot(y_pred, label="Predicted", fill=True, color="red")

plt.title("KDE plot of Actual vs Predicted Price")
plt.xlabel("Actual Price (blue)")
plt.ylabel("Predicted Price (red)")
plt.legend()

plt.show()

#=======================
# Ridge 
#=======================

#Ridge Regression Model
ridge = Ridge(alpha=10)

ridge.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridge.score(X_train, y_train)
test_score_ridge = ridge.score(X_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))

# Different alphas automatically
alphas = [0.01, 0.1, 1, 10, 50, 100, 200]
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X_train, y_train)

print("Best alpha:", ridge_cv.alpha_)
print("Train R²:", ridge_cv.score(X_train, y_train))
print("Test R²:", ridge_cv.score(X_test, y_test))


#=====================================
#Pipelines
#====================================

alphas = [0.01, 0.1, 1, 10, 100]

ridge_cv = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=alphas, store_cv_results=True)
)
ridge_cv.fit(X_train, y_train)

# Predictions on train and test set
y_pred_train = ridge_cv.predict(X_train)
y_pred_test = ridge_cv.predict(X_test)

# Predict on the test set
y_pred = ridge_cv.predict(X_test)

# Calculate metrics
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)


rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)


r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)


print("\n--- Ridge Regression with Cross-Validation ---")
print("Best alpha:", ridge_cv.named_steps['ridgecv'].alpha_)   # get best alpha
print(f"Train RMSE: {rmse_train:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")
print(f"Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")

#==========================
# Lasso Regression
#===========================

# Choose alpha
alpha = [0.01, 0.1, 1, 10, 100]

# Pipeline
lasso_pipe = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=alpha, cv=5)
)

# Fit the model
lasso_pipe.fit(X_train, y_train)

# Predict on the test set
y_pred_lasso = lasso_pipe.predict(X_test)

# Calculate metrics
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\n--- Lasso Regression with Cross-Validation ---")
print("Best alpha:", lasso_pipe.named_steps['lassocv'].alpha_)   # get best alpha
print(f"RMSE: {rmse_lasso:.4f}")
print(f"MAE: {mae_lasso:.4f}")
print(f"R²: {r2_lasso:.4f}")


#===================================
#Random Forest with Cross-validation
#===================================
rfr = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)

rfr.fit(X_train, y_train)

y_pred_rfr = rfr.predict(X_test)

mse_rfr  = mean_squared_error(y_test, y_pred_rfr)
rmse_rfr = np.sqrt(mse_rfr)
mae_rfr  = mean_absolute_error(y_test, y_pred_rfr)
r2_rfr   = r2_score(y_test, y_pred_rfr)

#===========================
# KFolds cross validatin for more stable metrics
#===========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse = cross_val_score(rfr, X, y, cv=kf, scoring="neg_mean_squared_error")
rmse_cv = np.sqrt(-neg_mse)

print(f"\nCross-validation RMSE per fold: {rmse_cv}")
print(f"Mean CV RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")

# Feature Importance
importances = rfr.feature_importances_
if hasattr(X_train, "columns"):
    feature_names = X_train.columns
else:
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]

feat_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\nTop 10 most important features:\n")
print(feat_importances.head(10))

# Plot top 10 features
plt.figure(figsize=(10,8))
feat_importances.head(10).plot(kind="barh", color="skyblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Features Importances (RandomForestRegressor)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()



#=========================================
#Linear regression
#====================================
lr = LinearRegression()

lr.fit(X_train, y_train)
y_lr_pred = lr.predict(X_test)

mse_liner_reg = mean_squared_error(y_test, y_lr_pred)
r2_linear_reg = r2_score(y_test, y_lr_pred)

print(f"\nMSE for Liner regression : {mse_liner_reg}")
print(f"R^2 score for linear regression : {r2_linear_reg}")

#=================================
# Hyperparameters and Elastic Net
#=================================

# Find the best alpha

lasso_pipe.named_steps["lassocv"].alpha_

ridge_cv.named_steps["ridgecv"].alpha_
#========================================
# Initialize Elastic Net Regression model
#========================================
from sklearn.linear_model import ElasticNet
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5)

# Fit the model on the training data
elastic_net_model.fit(X_train, y_train)

# Predict on the trest set
y_pred_elastic = elastic_net_model.predict(X_test)

# Calculate r^2 and MSE
r2_elastic = r2_score(y_test, y_pred_elastic)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)

print(f"Elastic Net Regression - Test MSE: {mse_elastic}")
print(f"Elastic Net Regression - Test R^2 Score: {r2_elastic}")


# Standartize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ElasticNet regresiiion with verifying best alpha
alpha_el=np.logspace(-3, 2, 50)
coefs = []
for alpha in alpha_el:
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000) # l1 ratio means equal weights for L1 and L2 penalties
    elastic_net.fit(X_train_scaled, y_train)
    coefs.append(elastic_net.coef_)

coefs= np.array(coefs)
print(f"\nCoeficients are : {coefs.shape}")

# Plotting the coefficient parths
plt.figure(figsize=(10,6))
for i in range(coefs.shape[1]):  # one curve per feature
    plt.plot(alpha_el, coefs[:, i], label=f"feature {i}")

plt.xscale("log")
plt.title("Elastic Net Regression coefficient paths")
plt.xlabel("alpha")
plt.ylabel("coefficient values")
plt.legend()
plt.grid(True)
plt.show()


#================================
# Grid Search
#================================
from sklearn.model_selection import GridSearchCV
ridge=Ridge()

param_grid = {"alpha" : [0.01,0.1,1,10,100]}
gridSearch = GridSearchCV(ridge, param_grid, cv=5, scoring="neg_mean_squared_error")

gridSearch.fit(X_train, y_train)

print(f"\nBest Ridge paramether is {gridSearch.best_params_}")
print(f"\nBest Ridge score is {gridSearch.best_score_}")
print("\nBest hiperparamether is ",gridSearch.best_estimator_)

y_pred_grid = gridSearch.predict(X_test)

print(f"\nClasification report of predicted grid search : \n{y_pred_grid}")

mse_grid = np.sqrt(mean_squared_error(y_test, y_pred_grid))
r2_grid = r2_score(y_test, y_pred_grid)

print(f"\nMean squared error for GridSearch is : {mse_grid}")
print(f"\nR^2 score for GridSearch is : {r2_grid}")


#======================
# Compare models
#=====================

# Lasso without Pipline
from sklearn.linear_model import Lasso

lasso = Lasso(max_iter=10000)

param_grid = {
    "alpha": [0.001, 0.01, 0.1, 1, 10]  # small alphas allow more features
}

lasso_search = GridSearchCV(lasso, param_grid, cv=5, scoring="neg_mean_squared_error")
lasso_search.fit(X_train, y_train)

print("\nBest Lasso alpha:", lasso_search.best_params_)

# Elastic Net 
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
elastic_net.fit(X_train_scaled, y_train)
y_pred_elastic = elastic_net.predict(X_test_scaled)

elastic_pipe = make_pipeline(StandardScaler(), ElasticNet(max_iter=10000))
param_grid = {"elasticnet__alpha":[0.001,0.01,0.1,1,10],
              "elasticnet__l1_ratio":[0.1,0.3,0.5,0.7,0.9]}
elastic_search = GridSearchCV(elastic_pipe, param_grid, cv=5, scoring="neg_mean_squared_error")
elastic_search.fit(X_train, y_train)

print("\nBest ElasticNet params:", elastic_search.best_params_)

# Ranfom Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [5, 10, 20],
    "max_features": [None, "sqrt", "log2", 1.0, 0.8],  # sin 'auto'
    "min_samples_split": [2, 5, 10],
}
rf_search = GridSearchCV(
    rf, param_grid, cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1, error_score="raise"
)

rf_search.fit(X_train, y_train)

print("Best RF params:", rf_search.best_params_)



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Function to evaluate and print results
def evaluate_model(model, X_train, y_train, X_test, y_test, name="Model"):
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test, y_test_pred))

    mae_test   = mean_absolute_error(y_test, y_test_pred)
    r2_test    = r2_score(y_test, y_test_pred)

    print(f"\n{name} Results:")
    print(f"  Train RMSE: {rmse_train:.4f}")
    print(f"  Test RMSE : {rmse_test:.4f}")
    print(f"  Test MAE  : {mae_test:.4f}")
    print(f"  Test R²   : {r2_test:.4f}")

    return {
        "model": name,
        "rmse_train": rmse_train,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "r2_test": r2_test
    }

# Example usage with best models from GridSearch
results = []
results.append(evaluate_model(gridSearch.best_estimator_, X_train, y_train, X_test, y_test, "Ridge (GridSearch)"))
results.append(evaluate_model(lasso_search.best_estimator_, X_train, y_train, X_test, y_test, "Lasso (GridSearch)"))
results.append(evaluate_model(elastic_search.best_estimator_, X_train, y_train, X_test, y_test, "ElasticNet (GridSearch)"))
results.append(evaluate_model(rf_search.best_estimator_, X_train, y_train, X_test, y_test, "RandomForest (GridSearch)"))

# Convert results to DataFrame for easier comparison
import pandas as pd
df_results = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(df_results)

