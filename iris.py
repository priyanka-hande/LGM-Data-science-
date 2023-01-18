import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
iris = pd.read_csv('/content/iriss.csv')
print(iris)
iris.head()
iris.shape
iris.describe(include='all')
iris.info()
iris.drop(columns="petal_width",inplace=False)
iris.isnull().sum()
iris.corr()
import seaborn as sns
plt.subplots(figsize = (7,5))
sns.heatmap(iris.corr(),annot=True,cmap="YlGnBu").set_title("Corelation of attributes on Iris species")
plt.show()
target=iris['species']
print(target)
target.value_counts()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features,target)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
from sklearn.naive_bayes import GaussianNB
#created a object for the classifier
bc=GaussianNB()
#train usinge train data
bc.fit(x_train,y_train)
pred=bc.predict(x_test)
print(pred)
from sklearn.metrics import classification_report
print(classification_report(pred,y_test))
from pandas.plotting import parallel_coordinates
parallel_coordinates(iris, "species")
iris.plot(kind="scatter", x="sepal_length", y="sepal_width")
sns.pairplot(iris)
