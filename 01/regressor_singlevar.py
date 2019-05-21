# 创建线性回归器
import sys
import numpy as np
filename = sys.argv[1] # 输入文件路径
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [
            float(i) for i in line.split(",")
        ]
        X.append(xt)
        Y.append(yt)

# 分割数据集,80%作为训练集，20%作为测试集
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training]).reshape((num_training, 1))
Y_train = np.array(Y[:num_training])

# 测试数据
X_test = np.array(X[num_training:]).reshape(num_test, 1)
Y_test = np.array(Y[num_training:])

# 创建一个回归器对象
from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()

# 用训练集训练模型
linear_regressor.fit(X_train, Y_train)

# 展示拟合效果
import matplotlib.pyplot as plt

y_train_pred = linear_regressor.predict(X_train) # 对训练集进行预测
plt.figure()
plt.scatter(X_train, Y_train, color='green')
#plt.scatter(X_train, y_train_pred, color='red')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title("Training Data")
plt.show()

# 模型对测试数据集进行预测
y_test_pred = linear_regressor.predict(X_test) # 对测试集进行预测

plt.scatter(X_test, Y_test, color='green')
#plt.scatter(X_train, y_train_pred, color='red')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title("Test Data")
plt.show()

## 计算回归准确性
import sklearn.metrics as sm

print("Mean absolute error:",round(sm.mean_absolute_error(Y_test, y_test_pred), 2))
print("Mean squred error:", round(sm.mean_squared_error(Y_test, y_test_pred), 2))
print("Median absolute error: ",round(sm.median_absolute_error(Y_test, y_test_pred), 2))
print("Explained variance score:", round(sm.explained_variance_score(Y_test, y_test_pred), 2))
print("R2 score:", round(sm.r2_score(Y_test, y_test_pred),2))

# 保存数据模型
import pickle

out_put_file = 'save_model.pkl'
with open(out_put_file, 'wb') as f:
    pickle.dump(linear_regressor, f)

# 使用数据模型
with open(out_put_file, 'rb') as f:
    model_linearead = pickle.load(f)

y_test_pred_new = model_linearead.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(Y_test, y_test_pred_new), 2))
