import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#-----------------------loading the dataset--------------------------------
df=pd.read_csv("heart.csv")
print(df)
#------------------------cleaning of the dataset-------------------
# checking the rows and the columns
print(df.shape)
# checking the first 5 rows
print(df.head())
#checking the last five rows
print(df.tail())
#checking the colunmns name 
print(df.columns)
# checking the datdtypes
print(df.dtypes)
# checking the non null values 
df.info()
# checking the null values in the columns 
print(df.isnull().sum())
# checking the duplicates
print(df[df.duplicated()])
# satitical summary of the dataset
print(df.describe())
# checking the unique values 
print("unique value per columns",df.nunique())
print("target value count",df["target"].value_counts())
#---------------rowise anlysis -----------------
print("first row",df.iloc[0])
print("first five rows",df.iloc[0:5])
# rowwise basic stat
df["row_sum"] =df.sum(axis=1)
df["row_mean"]=df.mean(axis=1)
print("added the rowwise sum and mean",df.head())
#--------------conditional analysis-------------
older_patients=df[df["age"]>55]
print("average target for older patients",older_patients["target"].mean())
print("overall avearge target",df["target"].mean())
#insights: the heart diease is less frequently in patients over 55 compared to overall populaation
#2) chest pain type impact
chest_pain =df[df["cp"]==3]
print("average dieases occurance for cp=3",chest_pain["target"].mean())
print("average diease occur ",df["target"].mean())
#insights:highter chest pain show the heart dieases symtoms
#3)compare the blood pressure
high_bp=df[df["trestbps"]>120]
print("avaerage target for the bp:",high_bp["target"].mean())
print("overalll avearge target",df["target"].mean())
#insights:high blood pressure correlated with higher disease probablity
#--------------------visualization----------------
plt.figure(figsize=(5,3))
sns.countplot(x="target",data=df,palette="coolwarm")
plt.title("haert disease distribution(0=no,1=yes)")
plt.show()
time.sleep(4)
# age distribution
plt.figure(figsize=(6,4))
sns.histplot(df['age'], bins=20 ,kde=True,color="skyblue")
plt.title("age distribution")
plt.xlabel("age")
plt.ylabel("Count")
plt.show()
time.sleep(3)
#cholestrol vs heart disease
plt.figure(figsize=(6,4))
sns.boxplot(x="target",y="chol",data=df,palette="pastel")
plt.title("heart disease vs choresoral")
plt.xlabel("haert disease(0=n0,1=yes)")
plt.ylabel("cholesterol")
plt.show()
time.sleep(3)
#age vs max heart rate
plt.figure(figsize=(6,4))
sns.scatterplot(x='age', y='thalach', hue='target', data=df, palette='Set1')
plt.title("Age vs Max Heart Rate (Colored by Heart Disease)")
plt.show()
time.sleep(3)
#--------------------coorealation and heat map
corr=df.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=False,cmap="coolwarm",linewidth=0.5)
plt.title("corelation haetmap")
plt.show()
print("corelation with target ",corr["target"].sort_values(ascending=False))
#-------------------model training------------
x=df.drop(columns=["target","row_sum","row_mean"])
y=df["target"]
# now splitting the data 
SEED=42
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=SEED)
# train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
feature_names = x.columns
coeff_df = pd.DataFrame(model.coef_[0], index=feature_names, columns=['Coefficient']).sort_values('Coefficient', ascending=False)
print("\nFeature Coefficients (sorted):\n", coeff_df)
# prdicting 
y_pred=model.predict(x_test)
# actual vs predicting
results=pd.DataFrame({
    "Actual":y_test.values,
    "Predicted":y_pred
})
print("actual vs predicted",results.head(10))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
#-------------confusion matrix-------
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d", cmap="Blues")
plt.title("confussion matrix")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()