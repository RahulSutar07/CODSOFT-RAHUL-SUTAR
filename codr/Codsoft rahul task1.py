import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
titanic_data = pd.read_csv(r"C:\Users\sutar\Downloads\Titanic-Dataset.csv")

# Exploratory Data Analysis
print(f"First few rows:\n{titanic_data.head()}\n")
print(f"Dataset shape: {titanic_data.shape}\n")
print(f"Descriptive statistics:\n{titanic_data.describe()}\n")
print(f"Survival counts:\n{titanic_data['Survived'].value_counts()}\n")

# Visualize survival counts by passenger class
plt.figure(figsize=(12, 8))  # Increase the figure size
sns.countplot(x='Survived', hue='Pclass', data=titanic_data, palette='viridis')
plt.title('Survival Counts by Passenger Class', fontsize=20, fontweight='bold')  # Increase font size and weight
plt.xlabel('Survival Status', fontsize=16, fontweight='bold')  # Increase font size and weight
plt.ylabel('Count', fontsize=16, fontweight='bold')  # Increase font size and weight
plt.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label size
plt.tight_layout()
plt.savefig('survival_counts_by_class.png', dpi=300, bbox_inches='tight')  # Save the figure

# Preprocess the data
titanic_data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
inputs = titanic_data.drop('Survived', axis=1)
target = titanic_data['Survived']

# One-hot encode categorical features
inputs = pd.get_dummies(inputs, columns=['Sex'])
inputs['Age'] = inputs['Age'].fillna(inputs['Age'].mean())  # Fill missing age values with mean

# Visualize survival counts by gender
survival_counts = titanic_data.groupby(['Survived', 'Sex']).size().unstack().fillna(0)
bar_width = 0.35  # Decrease bar width
index = np.arange(2)  # Use numeric indices for x-axis

fig, ax = plt.subplots(figsize=(12, 8))  # Increase the figure size
bar1 = ax.bar(index - bar_width/2, survival_counts['male'], bar_width, label='Male', color='#800080')
bar2 = ax.bar(index + bar_width/2, survival_counts['female'], bar_width, label='Female', color='#8B7D6B')
ax.set_xlabel('Survival Status', fontsize=16, fontweight='bold')  # Increase font size and weight
ax.set_ylabel('Count', fontsize=16, fontweight='bold')  # Increase font size and weight
ax.set_title('Survival Counts by Gender', fontsize=20, fontweight='bold')  # Increase font size and weight
ax.set_xticks(index)
ax.set_xticklabels(['Not Survived', 'Survived'], fontsize=14)  # Increase tick label size
ax.tick_params(axis='y', labelsize=14)  # Increase y-axis tick label size
ax.legend(fontsize=16, loc='upper right')  # Increase legend font size and change location
plt.tight_layout()
plt.savefig('survival_counts_by_gender.png', dpi=300, bbox_inches='tight')  # Save the figure

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")