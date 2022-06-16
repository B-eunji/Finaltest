from sklearn.datasets import load_wine 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# wine 데이터셋 읽어오기
wine = load_wine()

# feature, target 저장
wine_data = wine.data
wine_target = wine.target

# train데이터, test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, 
                                                    random_state = 0, train_size = 0.2)
#모델 선정
estimator = RandomForestClassifier()

#탐색 범위 정의
random_search = {'criterion': ['gini','entropy'],
               'min_samples_split':[None, 1,2,3,4],
               }

#r_search 정의
r_search = RandomizedSearchCV(estimator, random_search)
               

#r_search 학습
r_search.fit(x_train,y_train)

#결과 출력
test_score = r_search.score(x_test, y_test)
print("test set score: {}".format(test_score))

print("가장 최적의 파라미터 : {}".format(r_search.best_params_))
print("가장 최적의 스코어 값: {}".format(r_search.best_score_))