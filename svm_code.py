from sklearn.datasets import load_wine
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# wine 데이터셋 읽어오기
wine = load_wine()

# feature, target 저장
wine_data = wine.data
wine_target = wine.target

# train데이터, test 데이터 split
x_train, x_test, y_train, y_test = train_test_split(wine_data, wine_target, 
                                                    random_state = 0, train_size = 0.4)
# 모델 적용
svm_model = svm.SVC(C=1.5, gamma=1, kernel='linear')
svm_model.fit(x_train,y_train)

# 예측
pred = svm_model.predict(x_test)

# svm정확도 확인
svm_accuracy = accuracy_score(y_test, pred)
print("svm의 정확도: " ,svm_accuracy) 