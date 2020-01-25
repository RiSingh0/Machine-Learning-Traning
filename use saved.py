import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


daibetes=datasets.load_diabetes()
#(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print(daibetes.DESCR)
#daibetes_X = daibetes.data[:,np.newaxis,2]
daibetes_X = daibetes.data
daibetes_X_train = daibetes_X[:-30]
daibetes_X_test = daibetes_X[-30:]

daibetes_Y_tarin = daibetes.target[:-30]
daibetes_Y_test = daibetes.target[-30:]

with open('model_pickle','rb') as f:
    model=pickle.load(f)
    



daibetes_Y_predicted =model.predict(daibetes_X_test)

print("mean squred error is :",mean_squared_error(daibetes_Y_test, daibetes_Y_predicted))
print("weight",model.coef_)
print("intercept",model.intercept_)

