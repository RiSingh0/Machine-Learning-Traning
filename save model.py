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

model = linear_model.LinearRegression()

model.fit(daibetes_X_train,daibetes_Y_tarin)
with open('model_pickle','wb') as f:
    pickle.dump(model,f)

