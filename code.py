# python 3.7
# Scikit-learn ver. 0.23.2
from sklearn import datasets
# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

iris = datasets.load_iris()
digits = datasets.load_digits()
#print(iris)
"""
X, y = load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0).fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y)
"""
X, y = load_breast_cancer(return_X_y=True)
clf = RidgeClassifier().fit(X, y)
print(clf.score(X, y))
