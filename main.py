import os
os.environ["Keras_Backend"] = "plaidml.keras.backend"
import tkinter
import glob
from light_progress.commandline import ProgressBar
from keras.utils.np_utils import to_categorical
import cv2
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt
import learning


size = (100, 100)


def dir_clicked():
    root = tkinter.Tk()
    root.withdraw()
    s_dir = os.path.abspath(os.path.dirname(__file__))
    dir_path = filedialog.askdirectory(initialdir=s_dir)
    root.destroy()
    return dir_path


def plot(history, size_txt, epochs, batch_size, now_count, plot_count):
    # acc, val_accのプロット
    title = "Epochs:{0} Batch size:{1} Image size:{2} {3}".format(epochs, batch_size, size_txt, now_count + 1)
    save_name = "{0}-{1}-{2}-{3}".format(epochs, batch_size, size[0], now_count + 1)
    plt.plot(history.history["acc"], label="accuracy", ls="-", marker="o")
    print("1")
    print(history.history)
    plt.plot(history.history["val_acc"], label="val_accuracy", ls="-", marker="x")
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc="best")
    plt.savefig(save_name + ".png")
    plt.cla()


def data_load():
    folder = ['ayame', 'fubuki', 'subaru']  # 使用する画像のそれぞれのフォルダ
    default_dir = r'D:\SystemFiles\pictures\cnn'  # 上のフォルダの親ディレクトリ

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    print("学習データ読み込み")
    with ProgressBar(2100) as progress_bar:
        for index, name in enumerate(folder):
            dir_name = default_dir + "/train/" + name
            files = glob.glob(dir_name + "/*.png")
            for i, file in enumerate(files):
                img = cv2.imread(file)
                r_img = cv2.resize(img, size)
                x_train.append(r_img)
                y_train.append(index)
                progress_bar.forward()

    print("テストデータ読み込み")
    with ProgressBar(900) as progress_bar:
        for index, name in enumerate(folder):
            dir_name = default_dir + "/test/" + name
            files = glob.glob(dir_name + "/*.png")
            for i, file in enumerate(files):
                img = cv2.imread(file)
                r_img = cv2.resize(img, size)
                x_test.append(r_img)
                y_test.append(index)
                progress_bar.forward()

    x_train = np.array(x_train, dtype='float32')
    x_train /= 255
    x_test = np.array(x_test, dtype='float32')
    x_test /= 255
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # One-hotベクトルに変換
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 学習用データとテストデータ
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    return x_train, y_train, x_test, y_test, len(folder)


epochs = [20]
batch_size = [64]
count = 5

x_train, y_train, x_test, y_test, ran = data_load()

for i in epochs:
    for j in batch_size:
        for k in range(count):
            history, size_txt = learning.main(x_train, y_train, x_test, y_test, i, j, ran)
            plot(history, size_txt, i, j, k, count)
