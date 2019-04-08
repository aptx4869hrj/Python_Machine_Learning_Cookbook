import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

#创建实例二维数据，输入数据
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1],
        [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

#寻找最近邻的数量
num_neighbors = 3

#输入数据点，不包括在输入数据中
input_point = [2.6, 1.7]
input_point = np.array(input_point).reshape(1,2)

#画出数据点
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')

#建立最近邻模型，用输入数据训练该对象
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)

#计算输入点与输入数据中所有点的距离
distance, indices = knn.kneighbors(input_point)

#打印出k个最近邻，indices是一个已排序的数组
print("\nk nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank+1) + "-->", X[index])

# 画出最近邻点，并突出显示
plt.figure()
plt.scatter(X[:, 0], X[:, 1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:, 0], X[indices][0][:][:, 1],
            marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[:, 0], input_point[:, 1], marker='X', s=150, color='k', facecolors='none')

plt.show()