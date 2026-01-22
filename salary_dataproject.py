import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

df = pd.read_csv("salary_data.csv")
print(df.shape)
print(df.head())

graph = sns.histplot(data= df , x= 'Age' , y= 'YearsExperience')
graph.figure.suptitle("graph = sns.histplot(data= df , x= 'Age' , y= 'YearsExperience')")
graph.figure.show()
print(input("Wait for me...."))


X = df[['YearsExperience','Age']]
Y = df['Salary']

x_train , x_test , y_train , y_test = train_test_split( X , Y , random_state= 46)

model = LinearRegression()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)

mae = mean_absolute_error(y_pred , y_test)
print("Mean absolute error:" , mae)

mse = mean_squared_error(y_pred , y_test)
print("Mean squared error:" , mse)

r2 = r2_score(y_pred , y_test)
print("R2 score:" , r2)


