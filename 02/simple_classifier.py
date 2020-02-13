import numpy as np
import matplotlib.pyplot as plt

# 创建样本数据
X = np.array([[3, 1], [2, 5], [1, 8], [6, 4], [5, 2], [3, 5], [4, 7], [4, -1]])

# 为这些点分配标记
y = [0, 1, 1, 0, 0, 1, 1, 0]
# 按照类型标记把样本数据分成两类
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

print(class_0[:,1])
print(class_1)

# 使用散点图将数据画出来
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], color='black', marker='x')
# plt.show()

# 在两类数据间画一条分割线
line_x = range(10)
line_y = line_x

# 用数学公式 y = x 创建一条直线
plt.plot(line_x, line_y, color='black', linewidth=3)
plt.show()