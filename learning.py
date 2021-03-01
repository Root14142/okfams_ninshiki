# 画像ファイルの形式はpng
# 画像のサイズは統一してなくてもよい

# フォルダ指定
import os
os.environ["Keras_Backend"] = "plaidml.keras.backend"
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
import datetime

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

MAXLEN = 250
size = (100, 100)
size_txt = str(size[0]) + "x" + str(size[1])


def time():
    now = datetime.datetime.now()
    now_time = now.strftime("%H%M%S")
    return now_time


def main(x_train, y_train, x_test, y_test, epochs, batch_size, folder_len):
    # モデル
    model = Sequential()
    model.add(Conv2D(input_shape=(size[0], size[1], 3), filters=20, kernel_size=(5, 5), strides=(1, 1), padding="same",
                     activation="relu"))
    model.add(Conv2D(filters=20, kernel_size=(5, 5), strides=(1, 1), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(folder_len))
    model.add(Activation("softmax"))

    model.summary()

    # コンパイル
    model.compile(optimizer='sgd',
                  loss="categorical_crossentropy",
                  metrics=['accuracy'])

    # 学習
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # モデルを保存
    model.save("my_model.h5")

    # 汎化制度の評価・表示
    score = model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print('Loss:{0[0]}\nAccuracy:{0[1]}S'.format(score))

    return history, size_txt
