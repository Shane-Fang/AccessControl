import cv2
from face_train import Model
from personID import *


def OpenCamera():
    cap = cv2.VideoCapture(0)  # 設定影像來源為電腦鏡頭
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # 載入辨識人臉工具
    color = (0, 255, 0)  # 用來框臉的框框顏色
    model = Model()
    model.load_model(file_path='facemodel.h5')  # 載入模型facemodel.h5為訓練好的人臉辨識模型

    while True:
        ok, frame = cap.read()  # 讀取鏡頭影像 回傳ok, frame ok代表是否成功讀取鏡頭影像
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 影像RGB轉Gray

        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=5, minSize=(32, 32))  # 識別人臉
        if len(faceRects) > 0:  # 如果識別到人臉，則執行以下功能
            for faceRect in faceRects:
                x, y, w, h = faceRect  # 取得座標及長寬
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]  # 稍微放大一點

                faceID = model.face_predict(image)  # 將圖片輸入進模型，讓模型判斷

                name = getPersonByID(faceID)  # 用faceID去找出ID.csv檔對應的人名
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)  # 畫出框框

                if name:  # 判斷是否有名字

                    # 文字提示是誰
                    cv2.putText(frame, name,
                                (x + 30, y + 30),  # 座標
                                cv2.FONT_HERSHEY_SIMPLEX,  # 字型
                                1,  # 字號
                                (255, 0, 255),  # 顏色
                                2)  # 字的線寬

                else:
                    pass

        cv2.imshow("camera", frame)  # 開啟視窗
        c = cv2.waitKey(10)  # 每隔10毫秒掃描使用者是否下指令
        if c & 0xFF == ord('q'):  # 按下Q，離開迴圈，關閉程式
            break

    cap.release()  # 釋放鏡頭資源
    cv2.destroyAllWindows()


if __name__ == '__main__':
    OpenCamera()
