import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

from cleanEmployeeData import cleanData

train_x, test_x, train_y, test_y, pos_mapping = cleanData(random_state=1)

# 特征都是离散值，且不仅仅是0，1值，所以用多项式
model = MultinomialNB()

model.fit(train_x, train_y)
y_pred_train = model.predict(train_x)
y_pred_test = model.predict(test_x)

# 精准率、召回率，F1
print("Scores for train:")
print(classification_report(train_y, y_pred_train, target_names=pos_mapping))

print("Scores for test:")
print(classification_report(test_y, y_pred_test, target_names=pos_mapping))


