import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from cleanEmployeeData import cleanData

train_x, test_x, train_y, test_y, pos_mapping = cleanData(random_state=1)

# model = SVC(class_weight='balanced', random_state = 10)
# param_grid = {"C":[100,1000,1e4],"gamma":[0.0001,0.001,0.01,0.1,1]} # best C=1e4,gamma=0.001
# grid = GridSearchCV(model, param_grid, cv=5, verbose=2)
# grid.fit(train_x, np.ravel(train_y))
# print("The best parameters are %s with a score of %0.2f"
#        % (grid.best_params_, grid.best_score_))

model = SVC(C=100, gamma = 0.0001, class_weight='balanced', random_state = 10)
model.fit(train_x, train_y)

y_pred_train = model.predict(train_x)
y_pred_test = model.predict(test_x)

# 精准率、召回率，F1
print("Scores for train:")
print(classification_report(train_y, y_pred_train, target_names=pos_mapping))

print("Scores for test:")
print(classification_report(test_y, y_pred_test, target_names=pos_mapping))

