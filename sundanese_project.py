import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

df = pd.read_csv("sundanese_tweets.csv")
print(df.shape)
print(df.head())

graph = sns.countplot(x= 'emotion' , data= df)
graph.figure.suptitle("Class Distribution")
graph.figure.show()
print(input("Wait for me..."))

X = df['text']
Y = df['emotion']

vectorizer = TfidfVectorizer()
x_vec =vectorizer.fit_transform(X)

x_train , x_test , y_train , y_test = train_test_split( x_vec, Y , random_state= 46 , train_size= 0.2)
print("Train/Test split:" , x_train.shape[0] , x_test.shape[0])

scaler = StandardScaler(with_mean= False)
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

model = RandomForestClassifier()
model.fit(x_train , y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test , y_pred)
print("Accuracy score:" , round(acc , 3))
cr = classification_report(y_test , y_pred)
print("Classification report:" , cr)
cm = confusion_matrix(y_test , y_pred)
print("Confusion matrix:" , cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm , annot= True , fmt= 'd' )
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
print(input("Wait for me..."))



