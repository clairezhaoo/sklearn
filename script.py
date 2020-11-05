from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import neighbors,metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

standardScaler = StandardScaler()


wine = pd.read_csv("winequality-white.csv", sep=";")
print(wine.info())
bins = (2, 6.5, 8)
group_names = [0, 1]   # 0 is bad, 1 is good
wine["quality"] = pd.cut(wine["quality"], bins = bins, labels = group_names)
wine["quality"].unique()
label_quality = LabelEncoder()
wine["quality"] = label_quality.fit_transform(wine["quality"])
sns.countplot(wine["quality"])
plt.show()

"""
# KMeans
bc = datasets.load_breast_cancer()
x = scale(bc.data)
print(x)
y = bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=.2)
# create model
model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)         # clustering only trains labels, no y

predictions = model.predict(x_test)
labels = model.labels_
print("Labels: ", labels)
print("Predictions: ", predictions)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: ", accuracy)
print(pd.crosstab(y_train, labels))
bench_k_means(model, "1", x)
"""


"""
data = pd.read_csv("car.data")
#print(data.head())

X = data[[
"buying",
"maint",
"safety"
]].values
y = data[["class"]]

#print(X, y)

# convert data from string to int
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
#print(X)

label_mapping = {
"unacc": 0,
"acc": 1,
"good": 2,
"vgood": 3
}
y["class"] = y["class"].map(label_mapping)
y = np.array(y)
#print(y)


# KNN model
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

knn.fit(X_train, y_train)   # train the model
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)
print("Predictions: ", prediction)
print("Accuracy: ", accuracy)
"""


"""
#very accruate: .97-->1
# SVM
iris = datasets.load_iris()
X = iris.data
y = iris.target
classes =["Iris Setosa", "Iris Versiscolour", "Iris Virginica"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
model = svm.SVC()
model.fit(X_train, y_train)
print(model)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Predictions: ", predictions)
print("Accuracy: ", accuracy)
"""

"""
Attributes of wine dataset
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10)Color intensity
11)Hue
12)OD280/OD315 of diluted wines
13)Proline
"""
"""
wine = datasets.load_wine()
X = wine.data
print(X)
y = wine.target
#classes =["class_0", "class_1", "class_2"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
# use standard scaler
# X_train = standardScaler.fit_transform(X_train)
# X_test = standardScaler.transform(X_test)

model = svm.SVC()
model.fit(X_train, y_train)
print(model)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Predictions: ", predictions)
print("Accuracy: ", accuracy)
"""
