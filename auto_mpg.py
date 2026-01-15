import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------loading the datset-----
df=pd.read_csv("auto_mpg.csv")
print(df)
#------------------------cleaning of the csv----------------------
# checking the rows and columns
print(df.shape)
# checking the first five rows
print(df.head())
# checking the last five rows
print(df.tail())
#checking the columns in the dataset
print(df.columns)
# checking the data types
print(df.dtypes)
# checking the non null values in the data set 
df.info()
# checking the null values 
print(df.isnull().sum())
# checking the duplicate 
print(df[df.duplicated()])
#statical summary of the data set
print(df.describe())
# rename the data set
df.rename(columns={
    "mpg":"miles per gallons"
},inplace=True)
print(df.columns)
#-----------------------row wise anlyses -----------------------
print(df.iloc[0])
print(df.iloc[0:5])
#-------------- conditional anaysis ------------
# filter the car having more than 150 housepower anlyse how perfoam in term of miles per gallon
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
high_horsepower_cars = df[df['horsepower'] > 150]
print(high_horsepower_cars[['horsepower', 'miles per gallons']].head())
avg_mpg_high_hp = high_horsepower_cars['miles per gallons'].mean()
print("Average MPG for high horsepower cars:", avg_mpg_high_hp)
avg_mpg_overall = df['miles per gallons'].mean()
print("Overall average MPG:", avg_mpg_overall)
# filling the nan values as i have replace the question mark with nana now nan to median
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())


#2) filter the car based on the weights how it perfoam in term of miles per gallons
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['miles per gallons'] = pd.to_numeric(df['miles per gallons'], errors='coerce')
df = df.dropna(subset=['weight', 'miles per gallons'])
heavy_cars = df[df['weight'] > 3500]
print(heavy_cars[['weight', 'miles per gallons']].head())
avg_mpg_heavy = heavy_cars['miles per gallons'].mean()
print("Average MPG for heavy cars:", avg_mpg_heavy)
avg_mpg_overall = df['miles per gallons'].mean()
print("Overall average MPG:", avg_mpg_overall)
#3)comparing how many cars more than 5 cylinder compare in term of miles paer gallons 
df['cylinder']=pd.to_numeric(df['cylinders'],errors='coerce')
df['miles per gallons']=pd.to_numeric(df['miles per gallons'] ,errors='coerce')
df = df.dropna(subset=['cylinders', 'miles per gallons'])
high_cylinder_cars=df[df['cylinders']>5]
print(high_cylinder_cars[['cylinders','miles per gallons']].head())
avg_mpg_high_cylinder=high_cylinder_cars["miles per gallons"].mean()
print("average mpg for high cylinders",avg_mpg_high_cylinder)
print("overall average mpg",avg_mpg_overall)
#----------------data visualization -----------------------
# distribution of the miles per gallons
plt.figure(figsize=(8,6))
plt.hist(df["miles per gallons"],bins=20,color='lightblue',edgecolor='black')
plt.title("distribution of the miles per gallons")
plt.xlabel("miles per gallon")
plt.ylabel("frquency")
plt.show()
time.sleep(4)
#violin plot for miles per gallon across differnt cylinder 
plt.figure(figsize=(8,6))
sns.violinplot(x="cylinder",y="miles per gallons",data=df)
plt.title("miles per gallons acroos differnt cylinders")
plt.xlabel("number of cylinder")
plt.ylabel("miles per gallon")
plt.show()
time.sleep(5)
#scatter plot between the weight and the miles per gallon
plt.figure(figsize=(8,6))
sns.scatterplot(x="weight",y="miles per gallons",data=df)
plt.title("weight vs miles per gallon")
plt.xlabel("weight")
plt.ylabel("miles per gallon")
plt.show()
time.sleep(5)
# boxplot for housepower and miles per gallon 
plt.figure(figsize=(8,6))
sns.boxplot(x="horsepower",y="miles per gallons",data=df)
plt.title("mpg vs horsepower")
plt.xlabel("horsepower")
plt.ylabel("miles per gallon")
plt.show()
time.sleep(5)
# line plot between the aacelation vs model year 
acceleration_by_year=df.groupby("model year")["acceleration"].mean().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x="model year",y="acceleration",data=acceleration_by_year,marker="o")
plt.title("model year")
plt.ylabel("average accerlation")
plt.show()
time.sleep(4)
#---------------------correlation heatmap-------------------
corr=df[["acceleration","model year","miles per gallons"]].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True,cmap="coolwarm",linewidth=0.5)
plt.title("correalataion heatmap between accaerlation ,modelyear and mpg ")
plt.show()
# ---------------model training -----------------------------
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')
df['miles per gallons'] = pd.to_numeric(df['miles per gallons'], errors='coerce')
df = df.dropna()
x=df[['horsepower','weight','acceleration','model year','cylinder']]
y=df['miles per gallons']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
rmse=np.sqrt(mse)
r2=model.score(x_test,y_test)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² (Coefficient of Determination): {r2:.4f}")
# model coefficients
coefficients=pd.DataFrame(model.coef_,x.columns,columns=['coefficients'])
print(coefficients)
# plotting between the actual and prediction
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted Miles per Gallon')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.show()