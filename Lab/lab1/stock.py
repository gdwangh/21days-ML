import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("msft_stockprices_dataset.csv")

X = data[['High Price','Low Price', 'Open Price']]
y = data[['Close Price']]

train_X, test_X, train_y,test_y = train_test_split(X, y, test_size=0.3)

# 线性回归
regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)  # 训练模型
# print(regr.intercept_, regr.coef_)

y_pred_lr = regr.predict(test_X)

acc1 = (abs(y_pred_lr - test_y)/test_y <= 0.01).mean()
print("lineRegression:%.2f" % acc1)

# SVR
svr_rbf = SVR(kernel='rbf',C=1e4,gamma=0.001)
svr_rbf.fit(train_X, np.ravel(train_y))

y_pred_svr = svr_rbf.predict(test_X)
acc2 = (abs(y_pred_svr.reshape(-1,1) - test_y)/test_y <= 0.01).mean()
print("SVR:%.2f" % acc2)