import sys
import numpy as np

filename = sys.argv[1] # 输入文件路径
X = []
Y = []
with open(filename, 'r') as f:  #filename=data_multivar
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        Y.append(yt)
# 分割数据集,80%作为训练集，20%作为测试集
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# 训练数据
X_train = np.array(X[:num_training])
Y_train = np.array(Y[:num_training])
# 测试数据
X_test = np.array(X[num_training:])
Y_test = np.array(Y[num_training:])

# 岭回归
from sklearn import linear_model
import sklearn.metrics as sm
# 构建岭回归分类器   alpha参数控制回归器的复杂程度，趋于0时，岭回归器就是用普通二乘法
# 的线性回归器。如果希望模型对异常值不那么敏感，就把alpha的值设置得大一点
# 线性回归的主要问题是对数据值异常敏感
ridge_regressor = linear_model.Ridge(alpha=1, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, Y_train)

y_test_pred_ridge = ridge_regressor.predict(X_test)
print("Mean absolute error:", round(sm.mean_absolute_error(Y_test, y_test_pred_ridge), 2))
print("Mean squared error:", round(sm.mean_squared_error(Y_test, y_test_pred_ridge),2))
print("Median absolute error:", round(sm.median_absolute_error(Y_test, y_test_pred_ridge), 2))
print("Explain variance score:",round(sm.explained_variance_score(Y_test, y_test_pred_ridge), 2))
print("R2 score:",round(sm.r2_score(Y_test, y_test_pred_ridge), 2))

# 线性回归器
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, Y_train)

y_test_pred = linear_regressor.predict(X_test)
print("Mean absolute error:", round(sm.mean_absolute_error(Y_test, y_test_pred), 2))
print("Mean squared error:", round(sm.mean_squared_error(Y_test, y_test_pred),2))
print("Median absolute error:", round(sm.median_absolute_error(Y_test, y_test_pred), 2))
print("Explain variance score:",round(sm.explained_variance_score(Y_test, y_test_pred), 2))
print("R2 score:",round(sm.r2_score(Y_test, y_test_pred), 2))

# 多项式回归器
from sklearn.preprocessing import PolynomialFeatures
# 将多项式的次数的初始值设置为3
polynomial = PolynomialFeatures(degree=20)
# X_train_transformed 表示多项式形式的输入
X_train_transformed = polynomial.fit_transform(X_train)

datapoint = np.array([0.39, 2.78, 7.11]).reshape(1, -1)
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, Y_train)

print("\nLinear regression:", linear_regressor.predict(datapoint))
print("\nPolynomial regression:", poly_linear_model.predict(poly_datapoint))
