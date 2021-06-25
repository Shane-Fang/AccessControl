import random
from personID import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

from load_dataset import load_dataset, resize_image, IMAGE_SIZE


class Dataset:
    def __init__(self, path_name):
        # 訓練集
        self.train_images = None
        self.train_labels = None

        # 驗證集
        self.valid_images = None
        self.valid_labels = None

        # 測試集
        self.test_images = None
        self.test_labels = None

        # 資料集載入路徑
        self.path_name = path_name

        # 當前庫採用的維度順序
        self.input_shape = None

    # 載入資料集並按照交叉驗證的原則劃分資料集並進行相關預處理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=2):
        # 載入資料集到記憶體
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 當前的維度順序如果為'th'，則輸入圖片資料時的順序為：channels,rows,cols，否則:rows,cols,channels
        # 這部分程式碼就是根據keras庫要求的維度順序重組訓練資料集
        if K.image_data_format() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 輸出訓練集、驗證集、測試集的數量
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')

            # 我們的模型使用categorical_crossentropy作為損失函式，因此需要根據類別數量nb_classes將
            # 類別標籤進行one-hot編碼使其向量化，在這裡我們的類別只有兩種，經過轉化後標籤資料變為二維
            train_labels = to_categorical(train_labels, nb_classes)
            valid_labels = to_categorical(valid_labels, nb_classes)
            test_labels = to_categorical(test_labels, nb_classes)

            # 畫素資料浮點化以便歸一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # 將其歸一化,影象的各畫素值歸一化到0~1區間
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN網路模型
class Model:
    def __init__(self):
        self.model = None
        self.history = Callback()  # 模型訓練時紀錄的參數loss, acc等等

        # 建立模型

    def build_model(self, dataset, nb_classes=2):
        # 構建一個空的網路模型，它是一個線性堆疊模型，各神經網路層會被順序新增，專業名稱為序貫模型或線性堆疊模型
        self.model = Sequential()
        # 以下程式碼將順序新增CNN網路需要的各層，一個add就是一個網路層
        self.model.add(Convolution2D(32, 3, 3, padding='same',
                                     input_shape=dataset.input_shape))  # 1 2維卷積層
        self.model.add(Activation('relu'))  # 2 啟用函式層

        self.model.add(Convolution2D(32, 3, 3, padding='same'))  # 3 2維卷積層
        self.model.add(Activation('relu'))  # 4 啟用函式層
        self.model.add(Dropout(0.25))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 5 池化層
        self.model.add(Dropout(0.25))  # 6 Dropout層

        self.model.add(Convolution2D(64, 3, 3, padding='same'))  # 7  2維卷積層
        self.model.add(Activation('relu'))  # 8  啟用函式層

        self.model.add(Convolution2D(64, 1, 1))  # 9  2維卷積層
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))  # 11 池化層
        self.model.add(Dropout(0.25))  # 12 Dropout層

        self.model.add(Flatten())  # 13 Flatten層
        self.model.add(Dense(512))  # 14 Dense層,又被稱作全連線層
        self.model.add(Activation('relu'))  # 15 啟用函式層
        self.model.add(Dropout(0.5))  # 16 Dropout層
        self.model.add(Dense(nb_classes))  # 17 Dense層
        self.model.add(Activation('Softmax'))  # 18 分類層，輸出最終結果

        # 輸出模型概況
        self.model.summary()

    # 訓練模型
    def train(self, dataset, batch_size=32, nb_epoch=20, data_augmentation=True):
        sgd = SGD(learning_rate=0.01, decay=1e-6,
                  momentum=0.9, nesterov=True)  # 採用SGD+momentum的優化器進行訓練，首先生成一個優化器物件

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])  # 完成實際的模型配置工作

        # 不使用資料提升，所謂的提升就是從我們提供的訓練資料中利用旋轉、翻轉、加噪聲等方法創造新的
        # 訓練資料，有意識的提升訓練資料規模，增加模型訓練量
        if not data_augmentation:
            self.history = self.model.fit(dataset.train_images,
                                          dataset.train_labels,
                                          batch_size=batch_size,
                                          epochs=nb_epoch,
                                          validation_data=(dataset.valid_images, dataset.valid_labels),
                                          shuffle=True)
        # 使用實時資料提升
        else:
            # 定義資料生成器用於資料提升，其返回一個生成器物件datagen，datagen每被呼叫一
            # 次其生成一組資料（順序生成），節省記憶體，其實就是python的資料生成器
            datagen = ImageDataGenerator(
                featurewise_center=True,  # 是否使輸入資料去中心化（均值為0），
                samplewise_center=True,  # 是否使輸入資料的每個樣本均值為0
                featurewise_std_normalization=True,  # 是否資料標準化（輸入資料除以資料集的標準差）
                samplewise_std_normalization=True,  # 是否將每個樣本資料除以自身的標準差
                zca_whitening=False,  # 是否對輸入資料施以ZCA白化
                shear_range=0.5  # 資料像素按比例平移
             )

            # 計算整個訓練樣本集的數量以用於特徵值歸一化、ZCA白化等處理
            datagen.fit(dataset.train_images)

            # 利用生成器開始訓練模型
            print(dataset.train_labels)

            self.history = self.model.fit(datagen.flow(dataset.train_images, dataset.train_labels,
                                                       batch_size=batch_size),
                                          validation_steps=dataset.train_images.shape[0],
                                          epochs=nb_epoch,
                                          validation_data=(dataset.valid_images, dataset.valid_labels))

    MODEL_PATH = 'facemodel.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        # 模型評估
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f" % (self.model.metrics_names[0], score[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))

    # 識別人臉
    def face_predict(self, image):
        # 依然是根據後端系統確定維度順序
        if K.image_data_format() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):

            image = resize_image(image)  # 尺寸必須與訓練集一致都應該是IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))  # 與模型訓練不同，這次只是針對1張圖片進行預測
        else:

            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

            # 浮點並歸一化
        image = image.astype('float32')

        # 給出輸入屬於各個類別的概率，我們是二值類別，則該函式會給出輸入影象屬於0和1的概率各為多少
        result = self.model.predict(image)
        print(result[0])
        # 給出類別預測
        result = self.model.predict_classes(image)
        print(result[0])
        # 返回類別預測結果
        return result[0]

    def ShowLossAndAcc(self):
        # 繪製loss及acc圖表
        print([i * 100 for i in self.history.history['accuracy']])
        ImageName = "Loss and Acc.png"
        fig = plt.figure()
        plt.subplots_adjust(wspace=0, hspace=0.5)  # 設定子表格間距
        ax1 = fig.add_subplot(2, 1, 1)  # 分割圖表為上下兩格
        ax2 = fig.add_subplot(2, 1, 2)  # 分割圖表為上下兩格
        ax1.plot(self.history.history['loss'], label='loss')  # 繪製loss
        ax1.plot(self.history.history['val_loss'], label='val_loss')  # 繪製val_loss
        ax2.plot([i * 100 for i in self.history.history['accuracy']], c='r', label='accuracy')  # 繪製acc
        ax2.plot([i * 100 for i in self.history.history['val_accuracy']], label='val_accuracy')  # 繪製val_acc
        ax1.set_title('Model Loss')  # 標題
        ax2.set_title('Model Accuracy')  # 標題
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.xlabel('Epoch')  # 設定x軸名稱
        plt.ylabel('percentage')  # 設定y軸名稱
        plt.savefig(ImageName, dpi=200)


if __name__ == '__main__':
    nb_classes = getPersonNum()  # 取得人數，設定分類器的參數
    dataset = Dataset('./data/')
    dataset.load(nb_classes=nb_classes)  # 載入資料

    model = Model()

    # 先前新增的測試build_model()函式的程式碼
    model.build_model(dataset, nb_classes=nb_classes)

    # 測試訓練函式的程式碼
    model.train(dataset)
    model.save_model(file_path='facemodel.h5')
    model.evaluate(dataset)
    model.ShowLossAndAcc()
