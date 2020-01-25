#Importing Modules
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

#Loading Datasets
iris = datasets.load_iris()


#taking any one feature by numpy slicing
x = iris["data"][:,3:]

#if 2 then true else false
y = (iris["target"] == 2).astype(np.int)

# train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
print(clf.predict(([[2.6]])))
#new plot

x_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new, y_prob[:,1], "g-", label="verginica")
plt.show()
