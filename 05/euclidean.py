import numpy as np
import json

# 计算user1和user2的欧氏距离分数
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('User' + user1 + ' not present in the dataset')
    if user2 not in dataset:
        raise TypeError('User' + user2 + ' not present in the dataset')

    # 提取两个用户均评过分的电影
    rated_by_both = {}

    for item in dataset[user1]:
        if item in dataset[user2]:
            rated_by_both[item] = 1

    # 如果两个用户都没评分过，得分为0
    if len(rated_by_both) == 0:
        return 0
    # 对于每个共同评分，只计算平方和的平方根，并将该值归一化，使评分介于0-1之间
    squared_differences = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_differences.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_differences)))

if __name__=='__main__':
    data_file = 'movie_ratings.json'

    with open(data_file, 'r') as f:
        data = json.loads(f.read())

    user1 = 'John Carson'
    user2 = 'Michelle Peterson'

    print("\nEuclidean score:")
    print(euclidean_score(data, user1, user2))