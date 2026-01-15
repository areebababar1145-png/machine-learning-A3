import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1) LOAD DATA
# --------------------------------------------------
df =pd.read_csv("insurance.csv")
print(df)


# --------------------------------------------------
# 2) BASIC DATA CHECKS
# --------------------------------------------------
print("\nShape (rows, cols):", df.shape)
print("\nHead:\n", df.head())
print("\nTail:\n", df.tail())
print("\nColumns:\n", df.columns)
print("\nDtypes:\n", df.dtypes)
print("\nInfo:")
df.info()

# missing values
print("\nMissing values per column:\n", df.isnull().sum())

# duplicates
print("\nDuplicate rows:\n", df[df.duplicated()])

# statistical summary (only numeric)
print("\nDescribe (numeric):\n", df.describe())

# --------------------------------------------------
# 3) (OPTIONAL) DROP UNNECESSARY COLUMNS

# --------------------------------------------------
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
    print("\nAfter dropping Unnamed: 0 -> shape:", df.shape)

# --------------------------------------------------
# 4) RENAME COLUMNS (optional, just for readability)
# --------------------------------------------------
df.rename(columns={
    "bmi": "body_mass_index",
    "children": "num_children"
}, inplace=True)

print("\nColumns after rename:\n", df.columns)

# --------------------------------------------------
# 5) ROW-WISE ANALYSIS (like you did in Boston)
# --------------------------------------------------
print("\nFirst row:\n", df.iloc[0])
print("\nFirst 5 rows (iloc):\n", df.iloc[0:5])

# row-sum / row-mean don’t make super sense for mixed text+numbers,
# so we only apply on numeric columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df["row_sum"] = df[numeric_cols].sum(axis=1)
df["row_mean"] = df[numeric_cols].mean(axis=1)
print("\nData with row_sum / row_mean:\n", df.head())

# --------------------------------------------------
# 6) CONDITIONAL ANALYSIS (like river_houses / high_crime)
# --------------------------------------------------

# 1) smokers vs non-smokers
smokers = df[df["smoker"] == "yes"]
non_smokers = df[df["smoker"] == "no"]

print("\nAverage charges (smokers):", smokers["charges"].mean())
print("Average charges (non-smokers):", non_smokers["charges"].mean())
print("Overall average charges:", df["charges"].mean())
# insight: smokers pay WAY more

# 2) high BMI people
high_bmi = df[df["body_mass_index"] > 30]
print("\nHigh BMI sample:\n", high_bmi[["age", "body_mass_index", "charges"]].head())
print("Avg charges high BMI:", high_bmi["charges"].mean())

# 3) region-wise avg
print("\nAvg charges by region:\n", df.groupby("region")["charges"].mean())

# --------------------------------------------------
# 7) COLUMN-WISE ANALYSIS
# --------------------------------------------------
print("\nBody Mass Index head:\n", df["body_mass_index"].head())
print("\nDescribe BMI:\n", df["body_mass_index"].describe())
print("\nSmoker value counts:\n", df["smoker"].value_counts())
print("\nRegion value counts:\n", df["region"].value_counts())
print("\nSex value counts:\n", df["sex"].value_counts())

# --------------------------------------------------
# 8) VISUALIZATION
# --------------------------------------------------

# boxplot: smoker vs charges
plt.figure(figsize=(6,4))
sns.boxplot(x="smoker", y="charges", data=df)
plt.title("Charges vs Smoker (yes/no)")
plt.show()
time.sleep(3)

# scatter: BMI vs Charges (colored by smoker)
plt.figure(figsize=(6,4))
sns.scatterplot(x="body_mass_index", y="charges", hue="smoker", data=df)
plt.title("BMI vs Charges (smokers pay more)")
plt.show()
time.sleep(3)

# histogram: charges
plt.figure(figsize=(6,4))
plt.hist(df["charges"], bins=30, edgecolor="black")
plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

# --------------------------------------------------
# 9) CORRELATION HEATMAP
# (only numeric columns, otherwise it errors)
# --------------------------------------------------
corr = df.select_dtypes(include=["int64", "float64"]).corr(numeric_only=True)
plt.figure(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap (numeric only)")
plt.show()

print("\nCorrelation with charges (descending):\n")
print(corr["charges"].sort_values(ascending=False))

# --------------------------------------------------
# 10) PAIRPLOT (important columns)
# --------------------------------------------------
important_cols = [
    "charges",
    "age",
    "body_mass_index",
    "num_children"
]
sns.pairplot(df[important_cols])
plt.show()
time.sleep(3)

# --------------------------------------------------
# 11) MODEL TRAINING (Multiple Linear Regression)
# --------------------------------------------------
# target = charges
# features = age, sex, bmi, children, smoker, region
# but we must REMOVE the helper columns we created: row_sum, row_mean
X = df.drop(columns=["charges", "row_sum", "row_mean"])
y = df["charges"]

# We have categorical columns: sex, smoker, region
# we can one-hot encode with pandas get_dummies
X = pd.get_dummies(X, drop_first=True)

print("\nEncoded feature columns:\n", X.columns)
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# 12) TRAIN-TEST SPLIT
# --------------------------------------------------
SEED = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED
)

print("\nTrain / test shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# --------------------------------------------------
# 13) TRAIN MODEL
# --------------------------------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print("\nregressor.intercept_:", regressor.intercept_)
print("regressor.coef_ shape:", regressor.coef_.shape)

# --------------------------------------------------
# 14) COEFFICIENTS AS DATAFRAME (sorted)
# --------------------------------------------------
feature_names = X.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(
    data=model_coefficients,
    index=feature_names,
    columns=["Coefficient value"]
).sort_values("Coefficient value", ascending=False)

print("\nCoefficients by feature (sorted):\n")
print(coefficients_df)

# --------------------------------------------------
# 15) PREDICT
# --------------------------------------------------
y_pred = regressor.predict(X_test)

results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred
})
print("\nActual vs Predicted (first 15):\n")
print(results.head(15))

# --------------------------------------------------
# 16) METRICS
# --------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\nMean absolute error (MAE): {mae:.2f}")
print(f"Mean squared error (MSE):  {mse:.2f}")
print(f"Root MSE (RMSE):           {rmse:.2f}")

# manual R^2
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2_manual = 1 - ss_res / ss_tot
print(f"\nR² (manual): {r2_manual:.4f}")

# sklearn R^2
r2_sklearn = regressor.score(X_test, y_test)
print(f"R² (sklearn .score): {r2_sklearn:.4f}")
