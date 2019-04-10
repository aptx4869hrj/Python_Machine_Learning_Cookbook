import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets

#定义一个函数，原来源码中2.X版本得utilities包不存在了，读取文件需要换个方式
def load_data(input_file):
    a = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            a.append(data)

    return np.array(a)

#加载数据
input_file = 'data_nn_classifier.txt'
data = load_data(input_file)
X, y = data[:, :-1],data[:, -1].astype(np.int)

#print(X, y)

