# 빅분기 실기 준비
# 1. 분류/회귀 , target 변수, 평가지표 파악
# 2. 데이터 불러오기
# 3. EDA  ex) df.info() , df.describe() , df.isnull().sum()  , df.shape , df.head() , df.tail()  
# 4. 전처리1 encoder : LabelEncoder, one-hot pd.get_dummies()  <- Categorical 일 때
#            Scaler : Minmax , Robust 등  <- Numeric일 때

# 인코딩 : 라벨 인코딩, 원핫 인코딩
# 라벨 인코딩 : 포도 -> 1, 사고 -> 2 , 딸기 -> 3 로 인코딩
# 원핫 인코딩 : 




import os
os.getcwd()

import pandas as pd


train = pd.read_csv('mart_train.csv')
test = pd.read_csv('mart_test.csv')

train.shape
test.shape

# 변수 빼지마 (한 번 생각하게 되면서 시간도 뺏기고 잘 안됨)
# 이상치도 빼지마 (데이터 천개중 이상치 1~2개 있다고 큰게 안 바뀜.)
# 결측치가 있으면 평균 중앙값 최빈값으로 대체하기

target = train.pop('total')
train.head()


cols = ['rating']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])


cols = ['rating']
from sklearn.preprocessing import RobustScaler 
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])



print(train.shape , test.shape)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape, test.shape)



from sklearn.model_selection import train_test_split
X_tr , X_val , y_tr, y_val = train_test_split(train, target, test_size=0.2, random_state=2024)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)


# 모델 고민하지 말고 랜포 쓰면 됨.
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestRegressor()  # 분류문제라면 RandomForestClassifier 로 바꾸기만 하면 됨.
model.fit(X_tr, y_tr)
pred = model.predict(X_val)
pred.shape

from sklearn.metrics import mean_squared_error
print(mean_squared_error(pred, y_val)**0.5)  # 근데 랜포결과가 너무 값이 높으면 다른 모델도 생각해보기


pred = model.predict(test)
submit = pd.DataFrame({'pred':pred})
submit
submit.to_csv('submit.csv', index=False) # pred 1개만 제출하라고 했기 때문에 index=False 안하면 감점임.
sub = pd.read_csv('submit.csv')
sub






# 선형회귀
