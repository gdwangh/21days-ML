import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def accScore(y_true,y_pred, threshold=0.01):
    return (abs(y_pred - y_true) / y_true <= threshold).mean()

data = pd.read_csv("msft_stockprices_dataset.csv")

X = data[['High Price','Low Price', 'Open Price']]
y = np.ravel(data[['Close Price']])

train_X, test_X, train_y,test_y = train_test_split(X, y, test_size=0.3)
ErrorTolerance = 0.005 # 0.5%

# 线性回归
regr = linear_model.LinearRegression()
regr.fit(train_X, train_y)  # 训练模型
# print(regr.intercept_, regr.coef_)

y_pred_train_lr = regr.predict(train_X)
y_pred_test_lr = regr.predict(test_X)

# 计算准确性
acc1_train = accScore(train_y, y_pred_train_lr, ErrorTolerance)
acc1_test = accScore(test_y, y_pred_test_lr, ErrorTolerance)
print("lineRegression Acc rate (ErrorTolerance=%.3f): train=%.2f, test=%.2f" % (ErrorTolerance,acc1_train, acc1_test))


# SVR rbf,用GridSearchCV()调参
# param_grid = {"C":[100,1000,1e4],"gamma":[0.0001,0.001,0.01,0.1,1]} # best C=1e4,gamma=0.001
#
# # 自定义打分函数
# my_score = make_scorer(accScore, greater_is_better=True, needs_proba=False)
# scoring = {'myScore': my_score}
#
# grid = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, verbose=2, scoring=scoring,refit='myScore')
# grid.fit(train_X, np.ravel(train_y))
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

# SVR rbf
svr_rbf = SVR(kernel='rbf',C=10000, gamma=0.001)
svr_rbf.fit(train_X, np.ravel(train_y))

y_pred_train_svrrbf = svr_rbf.predict(train_X)
y_pred_test_svrrbf = svr_rbf.predict(test_X)

acc2_train_rbf = accScore(train_y, y_pred_train_svrrbf, ErrorTolerance)
acc2_test_rbf = accScore(test_y, y_pred_test_svrrbf, ErrorTolerance)
print("SVR rbf Acc rate (ErrorTolerance=%.3f): train=%.2f, test=%.2f" % (ErrorTolerance, acc2_train_rbf, acc2_test_rbf))
