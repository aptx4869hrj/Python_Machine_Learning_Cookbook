from sklearn.naive_bayes import GaussianNB
from logistic_regression import plot_classifier
import numpy as np

# 将数据放在X和y中
input_file = 'data_multivar.txt'

X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

# 建立一个朴素贝叶斯分类器
# GaussianNB 函数指定了正态分布朴素贝叶斯模型（Gaussian Naive Bayes model）
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# 计算分类器的准确性
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of the classifier =", round(accuracy, 10), "%")

# 画出数据点和边界
plot_classifier(classifier_gaussiannb, X, y)