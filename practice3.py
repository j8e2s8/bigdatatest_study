import pandas as pd
import numpy as np


train = pd.read_csv('mart_train.csv')
test = pd.read_csv('mart_test.csv')
train.head()
# target = train.pop('total')
train.head()
test.head()

train.info()  # rating 제외하고 다 범주형


df = pd.concat([train, test], axis=0)


df_dummy = pd.get_dummies(df, columns= df.columns.drop(['rating', 'total']), drop_first=True)

train = df_dummy.iloc[:len(train),:]
y = train.pop('total')
test = df_dummy.iloc[len(train):,:].drop(columns=('total'))


cols = ['rating']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols] = scaler.fit_transform(train[cols])
test[cols] = scaler.transform(test[cols])


from sklearn.model_selection import train_test_split, GridSearchCV

X_tr , X_val , y_tr, y_val = train_test_split(train, y, test_size=0.2, random_state=2024)
print(X_tr.shape, X_val.shape, y_tr.shape, y_val.shape)


# 모델 고민하지 말고 랜포 쓰면 됨.
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier 
model = RandomForestRegressor(random_state=42)  # 분류문제라면 RandomForestClassifier 로 바꾸기만 하면 됨.

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV 설정
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)

# 모델 훈련
grid_search.fit(X_tr, y_tr)

# 최적의 하이퍼파라미터와 최고 성능 출력
print("최적의 하이퍼파라미터:", grid_search.best_params_)
print("최고 성능:", grid_search.best_score_)

# 테스트 세트로 예측
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_val)




model.fit(X_tr, y_tr)
pred = model.predict(X_val)
pred.shape


from sklearn.metrics import mean_squared_error
print(mean_squared_error(pred, y_val)**0.5)  # 근데 랜포결과가 너무 값이 높으면 다른 모델도 생각해보기

# 414061.0886755025  # 랜포


print(mean_squared_error(y_pred, y_val)**0.5)  # 근데 랜포결과가 너무 값이 높으면 다른 모델도 생각해보기
# 408024.15521042264  # 랜포 grid search




# 선형
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_tr, y_tr)
pred2 = model.predict(X_val)

print(mean_squared_error(pred2, y_val)**0.5) 
# 414061.0886755025


# xgboost
from xgboost import XGBRegressor





pred = model.predict(test)
submit = pd.DataFrame({'pred':pred})
submit
submit.to_csv('submit.csv', index=False) # pred 1개만 제출하라고 했기 때문에 index=False 안하면 감점임.
sub = pd.read_csv('submit.csv')
sub