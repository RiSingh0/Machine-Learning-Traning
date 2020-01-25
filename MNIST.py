from sklearn.datasets import fetch_openml

#Fetching dataset
mnist = fetch_openml('mnist_784')
print(mnist)