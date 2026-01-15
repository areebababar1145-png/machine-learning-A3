import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#----------------loading the dataset-------------
df=pd.read_csv("drop.csv", sep=";")
print(df)
#---------------------cleaning of the data--------
# checking the rows and columns 
print(df.shape)
# checking the first 5 rows 
print(df.head())
# checking the last 5 rows 
print(df.tail())
# checking the columns 
print(df.columns)
# checking the datatype 
print(df.dtypes)
# checking the non null values 
df.info()
# checking the null values in the columns 
print(df.isnull().sum())
# checking the duplicate 
print(df[df.duplicated()])
# satiscial summary of the data set 
print(df.describe())
# cleaning the columns names 
df.columns=df.columns.str.strip().str.replace('"','').str.replace(' ','_')
print("Cleaned column names:")
print(df.columns)
# converting the object columns in to numeric 
 # converting the object columns into numeric (except Target)
for col in df.columns:
    if df[col].dtype == 'object' and col != 'Target':
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data types after possible conversion:")
print(df.dtypes)
#--- rowise analysis ------------------
print("first row",df.iloc[0])
print("first five rows",df.iloc[0:5])
# rowwise basic stat 
df_numeric = df.apply(pd.to_numeric, errors='coerce')
df["row_sum"] = df_numeric.sum(axis=1)
df["row_mean"] = df_numeric.mean(axis=1)

#--- conditional anlysis -------------
# 1) Older students dropout rate
older_student = df[df["Age_at_enrollment"] > 30]
print("Dropout rate for older students:",
      (older_student["Target"] == "Dropout").mean())
print("Overall dropout rate:",
      (df["Target"] == "Dropout").mean())


# 2) Students with unpaid fees (Debtor = 1)
debtors = df[df["Debtor"] == 1]
print("Dropout rate for students with unpaid fees:",
      (debtors["Target"] == "Dropout").mean())
print("Overall dropout rate:",
      (df["Target"] == "Dropout").mean())


# 3) Students with fewer approved subjects in 1st semester
poor_performance = df[
    df["Curricular_units_1st_sem_(approved)"] <
    df["Curricular_units_1st_sem_(approved)"].mean()
]
print("Dropout rate for low performance:",
      (poor_performance["Target"] == "Dropout").mean())
print("Overall dropout rate:",
      (df["Target"] == "Dropout").mean())


# 4) Dropout rate by course
print("Dropout rate by course:")
drop_rate_by_course = df.groupby("Course")["Target"].apply(
    lambda x: (x == "Dropout").mean()
).sort_values(ascending=False)
print(drop_rate_by_course)


# 5) Evening students vs overall
evening_students = df[df["Daytime/evening_attendance"] == 0]
print("Dropout rate for evening students:",
      (evening_students["Target"] == "Dropout").mean())
print("Overall dropout rate:",
      (df["Target"] == "Dropout").mean())
#------------visualization--------------
# dropout  distribution 
plt.figure(figsize=(6,4))
sns.countplot(x="Target",data=df,palette="viridis")
plt.title("distribution of target (dropout/graduate/enrollled)")
plt.xlabel("Target class")
plt.ylabel("Count")
plt.show()
time.sleep(3)
# drop out by age 
plt.figure(figsize=(8,5))
sns.boxplot(x="Target",y="Age_at_enrollment",data=df)
plt.title("Age at enrollment vs dropout status")
plt.xlabel("Traget")
plt.ylabel("age at enrollment")
plt.show()
time.sleep(4)
# academic perfoamance
plt.figure(figsize=(8,5))
sns.boxplot(x="Target", y="Curricular_units_1st_sem_(approved)", data=df)
plt.title("Academic Performance vs Dropout")
plt.xlabel("Target")
plt.ylabel("Approved Units (1st Semester)")
plt.show()
time.sleep(5)
# finaical status vs dropout
plt.figure(figsize=(6,4))
sns.countplot(x="Debtor", hue="Target", data=df, palette="coolwarm")
plt.title("Debtor Status vs Dropout")
plt.xlabel("Debtor (1 = Has unpaid fees)")
plt.ylabel("Count")
plt.show()
time.sleep(4)
# dropout by course 
drop_rate_by_course = (
    df.groupby("Course")["Target"]
      .apply(lambda x: (x == "Dropout").mean())
      .sort_values(ascending=False)
)

print(drop_rate_by_course)

plt.figure(figsize=(12,6))
drop_rate_by_course.plot(kind="bar")
plt.title("Dropout Rate by Course")
plt.xlabel("Course")
plt.ylabel("Dropout Rate")
plt.show()
time.sleep(3)
# ------- correaltion heatmap-------------
df.columns = (df.columns.astype(str)
              .str.replace('"','', regex=False)
              .str.replace("\t","", regex=False)
              .str.strip()
              .str.replace(" ", "_")
              .str.replace("/", "_")
              .str.replace("(", "")
              .str.replace(")", ""))

# encode target
le = LabelEncoder()
df["Target_num"] = le.fit_transform(df["Target"])
selected_cols = [
    "Target_num",
    "Age_at_enrollment",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Scholarship_holder",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade"
]
corr_drop = df[selected_cols].corr()
plt.figure(figsize=(8,6))
sns.heatmap(
    corr_drop,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)
plt.title("Clean Correlation Heatmap (Dropout Dataset)")
plt.show()
time.sleep(3)
# ------------------- model training-----------
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

le = LabelEncoder()
df["Target_num"]=le.fit_transform(df["Target"])
features = [
    "Age_at_enrollment",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Scholarship_holder",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade"
]
x=df[features]
y=df["Target_num"]
#------ train ing and test split 
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#------ scaling 
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm_model=SVC(kernel="rbf",C=1.0,gamma="scale")
svm_model.fit(X_train_scaled, y_train)
#  predication accuracy
y_pred=svm_model.predict(X_test_scaled)
print("accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))
# confusion matrix 
plt.figure(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title("SVM Confusion Matrix (Drop Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
