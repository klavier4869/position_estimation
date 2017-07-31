import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# 1つの入力から2クラス分類をするモデルにおいては
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
# 平均二乗誤差を最小化する回帰問題の場合
model.compile(optimizer='rmsprop',
              loss='mse')

# ダミーデータの作成
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# ラベルデータをカテゴリの1-hotベクトルにエンコードする
one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

# データを訓練用とテスト用で分ける
x_train, x_test = np.split(data, [int(len(data)*0.8)])
y_train, y_test = np.split(labels, [int(len(data)*0.8)])

# 各イテレーションのバッチサイズを32で学習を行なう
model.fit(x_train, y_train, epochs=10, batch_size=32)

score = model.evaluate(x_test, y_test, batch_size=128)
