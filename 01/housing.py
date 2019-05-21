import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# 通过接口直接获取数据（即housing.data里的数据）
housing_data = datasets.load_boston()
# 使用shuffle函数把数据的顺序打乱
X, Y = shuffle(housing_data.data, housing_data.target, random_state=7)
# 分割数据集
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], Y[:num_training]
X_test, y_test = X[num_training:], Y[num_training:]

# 拟合决策树回归模型
# 选择一个最大深度为4的决策树
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)
# 用带AdaBoost算法的决策树回归模型进行拟合
ad_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ad_regressor.fit(X_train, y_train)

# 评价决策树回归器的训练效果
y_pred_dt = dt_regressor.predict(X_test)
mes = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)

print("Decision Tree performance:")
print("Mean Squared error:", round(mes,2))
print("Explained Variance score:", round(evs, 2))

# 评价AdaBoost算法改善的效果
y_pred_ad = ad_regressor.predict(X_test)
mes = mean_squared_error(y_test, y_pred_ad)
evs = explained_variance_score(y_test, y_pred_ad)

print("AdaBoost performance:")
print("Mean Squared error:", round(mes,2))
print("Explained Variance score:", round(evs, 2))


# 画出特征的相对重要性
def plot_feature_importances(feature_importances, title, feature_names):
    # 将重要性值标准化
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # 将得分从高到低排序
    index_sorted = np.flipud(np.argsort(feature_importances))

    # 让X坐标轴上的标签居中显示
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # 画条形图
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree regressor', housing_data.feature_names)
plot_feature_importances(ad_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names)