# Support Vector Regression (SVR)

# ライブラリのインポート
# データ解析用ライブラリ
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# データセットのインポート
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# 訓練用とテスト用へのデータセットの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 特徴スケーリング
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

# 訓練データを使用したモデルの訓練
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# テストデータを使用した結果の予測
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# モデルの評価
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)