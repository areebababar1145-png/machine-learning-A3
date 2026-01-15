import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
#-------------------------load dataset -------------------------------------
df = pd.read_csv("Boston.csv")

print(df)

#--------------------- cleaning of data--------------------------------------
# checking the rows and the columns
print(df.shape)
# checking the first five rows
print(df.head())
# checking the last five rows
print(df.tail())
# checking the columns names 
print(df.columns)
# checking the datatype 
print(df.dtypes)
# checking the non null valves 
df.info()
# checking the missing values per columns
print(df.isnull().sum())
# checkinng the duplicates 
print(df[df.duplicated()])
#statical summary of the dataset
print(df.describe())
#--------------------drop the unnecessary columns------------
df = df.drop(columns=['Unnamed: 0'])
#---------------data after the cleaning ---------------
print(df.isnull().sum())
print(df.head())
print(df.shape)
# -----------------renaming the data set
df.rename(columns={
    'crim': 'crime_rate',
    'zn': 'residential_zone',
    'indus': 'industrial_area',
    'chas': 'near_charles_river',
    'nox': 'nitric_oxide',
    'rm': 'avg_rooms_per_house',
    'age': 'old_houses_percent',
    'dis': 'distance_to_employment_centers',
    'rad': 'highway_accessibility',
    'tax': 'property_tax_rate',
    'ptratio': 'pupil_teacher_ratio',
    'black': 'black_population_index',
    'lstat': 'lower_status_percent',
    'medv': 'median_home_value'
}, inplace=True)
print(df.columns)
#-------------------------rowise anlaysis---------------------
print(df.iloc[0])
#checking the muliple row 
print(df.iloc[0:5])
# anlyze each row statics
df['row-sum'] = df.sum(axis=1)
#rowise mean 
df['row-mean'] = df.mean(axis=1)
print(df.head())
#-------------------------conditional row anlysis ---------------
# To check if location advantage
river_houses=df[df["near_charles_river"]==1]
print(river_houses[["near_charles_river","median_home_value"]].head())
print("average price near the river:",river_houses["median_home_value"].mean())
print("overall avearge price:",df["median_home_value"].mean())
#insights: we can say that the houses near the river cost more in this datset 
# near charles river show house near the water are more valuable

#2)To see how crime affects property prices.
high_crime=df[df["crime_rate"]>10]
print(high_crime[["crime_rate","median_home_value"]].head())
print("average price of high crime area ",high_crime['median_home_value'].mean())
print("overall average price",df["median_home_value"].mean())
#insights: the crime increase the median_home_value decreases
# when the median_home_value is lower than overall average this show me the negative relationship between them 

#3) test the bigger homes sell for the higher prices
large_houses = df[df["avg_rooms_per_house"] > 7]
print(large_houses[["avg_rooms_per_house","median_home_value"]].head())
print("average price o ",large_houses['median_home_value'].mean())
print("overall average price",df["median_home_value"].mean())
#insights: the houses with the more rooms trend to be more expensive 
 
 #---------------------------columns wise anlysis-----------
 # insepecting the single columns
print(df['crime_rate'].head())
print(df.loc[:"crime_rate"].describe())
print(df.iloc[:,0].describe())
# multiple columns selection 
block_loc = df.loc[50:60, ['crime_rate', 'avg_rooms_per_house', 'median_home_value']]
subset_pos =df.iloc[:,[0,5,-1]]
# staticus on the columns
print(df.describe())
# value counts from the categorical columns 
print(df["near_charles_river"].value_counts())
 
#--------------------visualization--------------
plt.figure(figsize=(6,4))
sns.boxplot(x='near_charles_river',y="median_home_value",data=df)
plt.title("home prices vs near charles river(0=no,1=yes)")
plt.show()
time.sleep(5)

# scatter plot 
plt.figure(figsize=(6,4))
sns.scatterplot(x="crime_rate",y="median_home_value",data=df)
plt.title("crome rate vs median house value")
plt.xlabel("crime rate")
plt.ylabel("house price")
plt.show()
time.sleep(9)

# histogram 
plt.figure(figsize=(6,4))
plt.hist(df["avg_rooms_per_house"], bins=25, color='lightgreen', edgecolor='black')
plt.title("Distribution of Average Rooms per Dwelling")
plt.xlabel("Average Number of Rooms")
plt.ylabel("Frequency")
plt.show()

# ------------------------------correlation---------------------------

corr=df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=False,cmap="coolwarm",linewidth=0.5)
plt.title("correalation haetmap")
plt.show()
print(corr["median_home_value"].sort_values(ascending=False))

#------------------pair plot----------------
important_cols = [
    "median_home_value",
    "avg_rooms_per_house",
    "lower_status_percent",
    "crime_rate",
    "pupil_teacher_ratio"
]

sns.pairplot(df[important_cols])
plt.show()
time.sleep(6)
#-------------------model training---------------------------------------
x=df.drop(columns=["median_home_value","row-sum","row_mean"])
y=df["median_home_value"]
# now doing the train test split
SEED=42
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=SEED)
#now training of the model
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("regressor.intercept",regressor.intercept_)
print("regressor.coef",regressor.coef_)
feature_names = x.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(
    data=model_coefficients,
    index=feature_names,
    columns=["Coefficient value"]
).sort_values("Coefficient value", ascending=False)

print("\nCoefficients by feature (sorted):\n")
print(coefficients_df)
#predicting
y_pred=regressor.predict(x_test)
result=pd.DataFrame({
    "actual":y_test.values,
    "predicted":y_pred
})
print("actual vs prdicted")
print(result.head(15))
# metices
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(f"\nMean absolute error (MAE): {mae:.2f}")
print(f"Mean squared error (MSE):  {mse:.2f}")
print(f"Root MSE (RMSE):           {rmse:.2f}")
# r^2 manaual
ss_res=np.sum((y_test-y_pred)**2)
ss_tot=np.sum((y_test-y_test.mean())**2)
r2_manaual=1-ss_res /ss_tot
print(f"\nR² (manual): {r2_manaual:.4f}")
#r2 sklearn
r2_sklearn = regressor.score(x_test, y_test)
print(f"R² (sklearn .score): {r2_sklearn:.4f}")
