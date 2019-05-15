#数据预处理

import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9, -4.3]])

#均值移除（Mean removal）
data_standardized = preprocessing.scale(data)
print("\nMean =", data_standardized.mean(axis=0))  #特征值为0
print("Std deviation =",data_standardized.std(axis=0))  #标准差为1

#范围缩放（Scaling）
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data =", data_scaled)

#归一化（Normalization）
#最常用的归一化形式是将特征向量调整为L1范数，使特征向量的数值之和为1
data_normalized = preprocessing.normalize(data, norm='l1')
print("\nL1 normalized data =",data_normalized)

#二值化（Binarization）
#用于将数值特征向量转换为布尔类型向量
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print("\nBinarized data =",data_binarized)

#独热编码
#one-of-k 的形式对每个值进行编码
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12],[1, 3, 5, 3],[2, 3, 2, 12],[1, 2, 4, 3]])
encoder_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded vector =", encoder_vector)

#Encoded vector = [[0. 0. 1. 0. 1. 0. 0. 0. 1. 1. 0.]]
#说明：以列为单位进行非重复计数和编码，以上面例子为例
#对[2, 3, 5, 3]中的2进行编码，先遍历原先的四维矩阵中对应的列向量[0, 1, 2, 1]
#1为重复数，所以重新编号排序后为[0， 1， 2]，那么[2, 3, 5, 3]中的2对应的编码就为0，0，1.
#以此类推
