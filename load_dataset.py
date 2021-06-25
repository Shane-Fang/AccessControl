import os
import numpy as np
import cv2
from personID import *

IMAGE_SIZE = 64


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 獲取影象尺寸
    h, w, _ = image.shape

    # 對於長寬不相等的圖片，找到最長的一邊
    longest_edge = max(h, w)

    # 計算短邊需要增加多上畫素寬度使其與長邊等長
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

        # RGB顏色
    BLACK = [0, 0, 0]

    # 給影象增加邊界，是圖片長、寬等長，cv2.BORDER_CONSTANT指定邊界顏色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 調整影象大小並返回
    return cv2.resize(constant, (height, width))


# 讀取訓練資料
images = []
labels = []


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 從初始路徑開始疊加，合併成可識別的操作路徑
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是資料夾，繼續遞迴呼叫
            read_path(full_path)
        else:  # 檔案
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                # ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
                images.append(image)
                labels.append(path_name)

    return images, labels


# 從指定路徑讀取訓練資料
def load_dataset(path_name):
    images, labels = read_path(path_name)

    # 將輸入的所有圖片轉成四維陣列，尺寸為(圖片數量*IMAGE_SIZE*IMAGE_SIZE*3)
    # 圖片為64 * 64畫素,一個畫素3個顏色值(RGB)
    images = np.array(images)

    Persons, ID = getPersonAndID()  # 取得人名和ID

    # 標記資料
    for i in range(len(labels)):
        for j in range(len(Persons)):
            if labels[i].endswith(Persons[j]):  # 如果labels路徑最後一個資料夾與ID.csv檔的人名一樣，則用ID進行標記
                labels[i] = ID[j]

    labels = np.array(labels)

    return images, labels


if __name__ == '__main__':
    images, labels = load_dataset("data")
    print(labels.shape)
