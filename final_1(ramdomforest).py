import pandas as pd
from sklearn.datasets import load_wine 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# wine 데이터셋 읽어오기
wine = load_wine()

#
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

wine_data = wine.data
wine_target = wine.target

x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, 
                                                    random_state = 0, train_size = 0.2)

estimator = RandomForestClassifier(criterion = 'entropy',
                                   max_depth = None,
                                   max_leaf_nodes = None,
                                   min_samples_split = 3,
                                   min_samples_leaf=1
                                   )

estimator.fit(x_train, y_train)

r_predict = estimator.predict(x_train)
score = metrics.accuracy_score(y_train, r_predict)
print(score)

r_predict = estimator.predict(x_test)
score = metrics.accuracy_score(y_test, r_predict)
print(score)