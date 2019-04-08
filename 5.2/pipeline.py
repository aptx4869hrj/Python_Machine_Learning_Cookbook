from sklearn.datasets import samples_generator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

#生成样本数据，一个20维的特征向量
X, y = samples_generator.make_classification(n_informative=4, n_features=20, n_redundant=0, random_state=5)

#特征选择器，选择k个最好的特征
selector_k_best = SelectKBest(f_regression, k=10)

#随机森林分类器分类数据
classifier = RandomForestClassifier(n_estimators=50, max_depth=4)

#构建机器学习流水线，pipeline方法允许我们用预定义的对象来创建流水线，并对模块进行制定命名
pipeline_classifier = Pipeline([('selector', selector_k_best), ('rf', classifier)])

#可以进行参数更新
pipeline_classifier.set_params(selector__k=6, rf__n_estimators=50)

#训练分类器
pipeline_classifier.fit(X, y)

#为训练数据预测输出结果
prediction = pipeline_classifier.predict(X)
print("\nPredictions:\n", prediction)

#打印分类器得分
print("\nScore:", pipeline_classifier.score(X, y))

#打印被分类器选择的特征
features_status = pipeline_classifier.named_steps['selector'].get_support()
selected_features = []
for count, item in enumerate(features_status):
    if item:
        selected_features.append(count)

print("\nSelected features (0-indexed):", ', '.join([str(x) for x in selected_features]))