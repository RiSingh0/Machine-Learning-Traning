#import require module
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading datasets
iris = datasets.load_iris()
#printing description
print(iris.DESCR)
#ready feacture and lables 
features = iris.data
lables= iris.target
#just printing 1st feature and lables
print(features[0],lables[0])

#Training the Classifier
clf = KNeighborsClassifier()
clf.fit(features,lables)

preds = clf.predict([[5.1,3.5,1.4,0.2]])
print(preds)

