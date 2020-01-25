import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error


daibetes=datasets.load_diabetes()
#(['data', 'target', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print(daibetes.DESCR)
# daibetes_X = daibetes.data[:,np.newaxis,2]


daibetes_X = daibetes.data
daibetes_X_train = daibetes_X[:-30]
daibetes_X_test = daibetes_X[-30:]

daibetes_Y_tarin = daibetes.target[:-30]
daibetes_Y_test = daibetes.target[-30:]

model = linear_model.LinearRegression()
print(daibetes_X_train)
print(daibetes_Y_tarin)
model.fit(daibetes_X_train,daibetes_Y_tarin)


daibetes_Y_predicted =model.predict(daibetes_X_test)
print("pridict",daibetes_Y_predicted)

print("mean squred error is :",mean_squared_error(daibetes_Y_test, daibetes_Y_predicted))
print("weight",model.coef_)
print("intercept",model.intercept_)
# print("pridict",daibetes_Y_predicted)
# plt.scatter(daibetes_X_test,daibetes_Y_test)
# plt.plot(daibetes_X_test,daibetes_Y_predicted)

# plt.show()

#mean squred error is : 3035.0601152912695
#weight [941.43097333]
#intercept 153.39713623331698

