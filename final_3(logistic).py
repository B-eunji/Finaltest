import numpy as np 
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

dataset = datasets.load_wine()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

x_data = dataset.data
y_data = dataset.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data,
                                                                    y_data,
                                                                    test_size
                                                                    =0.5)

estimator = LogisticRegression()

random_searh = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

r_search = RandomizedSearchCV(estimator, random_searh)

r_search.fit(x_train, y_train)

test_score = r_search.score(x_test, y_test)
print("test set score: {}".format(test_score))

print("가장 최적의 파라미터: {}".format(r_search.best_params_))
print("가장 최적의 스코어 값: {}".format(r_search.best_score_))