import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

data_path = r'C:\Users\sutar\Downloads\Iris.csv'
df = pd.read_csv(data_path)

print("First five rows of the dataset:")
print(df.head())

df = df.drop(columns=['Id'])
print("\nDataset after dropping 'Id' column:")
print(df.head())

print("\nDescriptive statistics:")
print(df.describe())

print("\nData types and null value summary:")
print(df.info())
print("\nNull value count:")
print(df.isnull().sum())

print("\nUnique values in 'Species' column:")
print(df['Species'].value_counts())

plt.figure(figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)

plt.subplot(2, 2, 1)
df['SepalLengthCm'].hist()
plt.title('Sepal Length Distribution')

plt.subplot(2, 2, 2)
df['SepalWidthCm'].hist()
plt.title('Sepal Width Distribution')

plt.subplot(2, 2, 3)
df['PetalLengthCm'].hist()
plt.title('Petal Length Distribution')

plt.subplot(2, 2, 4)
df['PetalWidthCm'].hist()
plt.title('Petal Width Distribution')

plt.show()

colors = ['red', 'orange', 'blue']
species = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']

for i, s in enumerate(species):
    subset = df[df['Species'] == s]
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2, 2, 1)
    plt.scatter(subset['SepalLengthCm'], subset['SepalWidthCm'], c=colors[i], label=s)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")
    plt.title(f'Sepal Length vs Sepal Width for {s}')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], c=colors[i], label=s)
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(f'Petal Length vs Petal Width for {s}')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(subset['SepalLengthCm'], subset['PetalLengthCm'], c=colors[i], label=s)
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.title(f'Sepal Length vs Petal Length for {s}')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.scatter(subset['SepalWidthCm'], subset['PetalWidthCm'], c=colors[i], label=s)
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.title(f'Sepal Width vs Petal Width for {s}')
    plt.legend()

    plt.show()

corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
print("\nDataset with encoded target variable:")
print(df.head())

from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier()
]

model_names = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Decision Tree"
]

for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test) * 100
    print(f"{name} Accuracy: {accuracy:.2f}%")