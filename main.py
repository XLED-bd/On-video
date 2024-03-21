import tensorflow as tf
import numpy as np
import cv2 as cv
from ultralytics import YOLO
#from easyocr import Reader

from conver import convert_matrix_to_vector

# /bin/python3.10 /home/ivan/project/python/neural_netwok/number_plate_yolo/On-video/main.py

car_detect = YOLO("yolov8n.pt")
plate_detect = YOLO("trained_plate_yolov8s.pt")
reader =tf.keras.models.load_model("big_ocr_model_42_128_57121_9_epoch.h5") # Reader(["en"], gpu=True)

def main():
    cap = cv.VideoCapture("/home/ivan/Videos/Записи экрана/Запись экрана от 17.03.2024 20:47:26.webm")

    clases = [2, 3, 5, 7]
    
    ret = True
    while ret:
        ret, orig_frame = cap.read()

        frame = cv.resize(orig_frame, (1280, 720))
        pred = car_detect.predict(frame)[0]
        for detect in pred.boxes.data.tolist():
            minx, miny, maxx, maxy, score, class_id = detect
            if class_id in clases and score > 0.6:
                frame = cv.rectangle(frame, (int(minx), int(miny)),
                        (int(maxx), int(maxy)), (0, 0, 255), 2)
                #car_frame = frame[int(miny):int(maxy), int(minx):int(maxx)] 
                pred2 = plate_detect.predict(frame, imgsz=800)[0]
                if pred2.boxes.data.tolist() != []:
                    #print(pred2.boxes.data.tolist())
                    minx2, miny2, maxx2, maxy2, score2, _ = pred2.boxes.data.tolist()[0]
                    if score2 > 0.5:
                        frame = cv.rectangle(frame, (int(minx2), int(miny2)),
                                        (int(maxx2), int(maxy2)), (255, 0, 0), 1)

                        plate_frame = frame[int(miny2):int(maxy2), int(minx2):int(maxx2)]
                        plate_frame = cv.cvtColor(plate_frame, cv.COLOR_BGR2GRAY)

                        
                        plate_frame = cv.resize(plate_frame, (128, 42))
                        #plate_frame = cv.threshold(plate_frame, 60, 255, cv.THRESH_BINARY)
                        cv.imwrite("plate_frame.png", plate_frame)

                        pre = np.zeros((1, 42, 128))
                        pre[0] = plate_frame
                        
                        answer = reader.predict(pre)
                        answer = convert_matrix_to_vector(answer)
                        a = ""
                        for i in answer:
                            a += i

                        cv.putText(frame, a, (int(minx2), int(miny2)), cv.FONT_HERSHEY_PLAIN, 2.3,
                                (255, 0, 0), 3, cv.LINE_AA)
                        

                        '''answer = reader.readtext(plate_frame, allowlist='ABMXPTYEHCO1234567890')

                        if answer != []:
                            a = ""
                            print(answer)
                            for word in answer:
                                a += word[1]
                            cv.putText(frame, a, (int(minx2), int(miny2)), cv.FONT_HERSHEY_PLAIN, 2.3,
                                (255, 0, 0), 3, cv.LINE_AA)'''
                        

        cv.imshow("123", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()