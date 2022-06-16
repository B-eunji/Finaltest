from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# wine 데이터셋 읽어오기
wine = load_wine()

# feature, target 저장
wine_data = wine.data
wine_target = wine.target

# train데이터, test 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, 
                                                    random_state = 0, train_size = 0.2)

#탐색 범위 정의
random_search = {
               'C': [0.1, 0.5, 1.0, 1.5, 2.0, 10, 100],
               'gamma': [0.1, 0.5, 1.0, 1.5, 2.0, 10, 100],
               'kernel': ['linear', 'sigmoid', 'rbf'] 
               }


#RandomizedSearchCV를 이용한 객체 생성
random_search = RandomizedSearchCV(SVC(), random_search, cv=5)

#random_search 모델 학습
random_search.fit(x_train,y_train)

#결과 출력
print("test set score: {}".format(random_search.score(x_test, y_test)))

print("가장 최적의 파라미터 : {}".format(random_search.best_params_))
print("가장 최적의 스코어 값: {}".format(random_search.best_score_))