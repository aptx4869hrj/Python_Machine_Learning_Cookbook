import csv
import sys
from sklearn.utils import shuffle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
#from housing import plot_feature_importances
from sklearn.metrics import mean_squared_error, explained_variance_score


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'r'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])

    # 提取特征名称
    feature_names = np.array(X[0])

    # 将第一行特征名称移除，仅保留数值
    X = np.array(X[1:]).astype(np.float32)
    y = np.array(y[1:]).astype(np.float32)
    return X, y, feature_names
    # 将第一行特征名称移除，仅保留数值
    return X, y, feature_names


# 读取数据，打乱顺序
filename = 'bike_day.csv'
X, y, feature_names = load_dataset(filename)
X, y = shuffle(X, y, random_state=7)
# 分割数据集
num_training = int(0.9 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 训练回归器
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
# n_estimators表示评估器数量，即随机森林需要使用的决策树的数量
# max_depth是最大深度；min_samples_split值决策树分裂一个节点需要用到的最小数据样本量，默认是2
rf_regressor.fit(X_train, y_train)

# 评价随机森林回归器的训练效果
y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\n#### Random Forest regressor performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

from housing import plot_feature_importances
plot_feature_importances(rf_regressor.feature_importance_, 'Random Forest regressor', feature_names)

