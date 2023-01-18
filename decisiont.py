import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:\Users\LENOVO\Documents")
df.head()
df.columns
df.isnull().sum()
df.describe()
df.corr()
df.groupby('species').size()
fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('equal')
colors = ['yellow','blue','green']
sp = df['species'].unique()
ct = df['species'].value_counts().tolist()
ax.pie(ct, labels = sp, autopct='%1.2f%%', colors=colors, shadow=True, startangle=90)
plt.show()
sns.pairplot(df, hue='species', height=3)
plt.show()
X = df.drop(['species'], 1)
y = df['species']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
X_train.head()
y_train.head()
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy", random_state=20)
classifier.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report: \n\n{classification_report(y_test, y_pred)}")
features = df.columns[:-1]
classes = df['species'].unique().tolist()
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 15))
plot_tree(classifier, feature_names=features, class_names=classes, filled=True)
plt.show()
