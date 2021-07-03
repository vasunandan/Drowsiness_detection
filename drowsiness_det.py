import cv2
from keras import models
import numpy as np
from pygame import mixer
import time

model = models.load_model('models/model_eye_detector.h5')

face_cascade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_alt.xml')
l_eye_cascade = cv2.CascadeClassifier('Resources/haarcascade_lefteye_2splits.xml')
r_eye_cascade = cv2.CascadeClassifier('Resources/haarcascade_righteye_2splits.xml')

# img = cv2.imread("Resources/messi_2.jpg")
# imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# face = l_eye_cascade.detectMultiScale(imgGray,1.1,4)

mixer.init()
sound = mixer.Sound('alarm.wav')

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,400)
cap.set(10,100)

l_pred = [-1]
r_pred = [-1]
# variables
sleep_count = 0
count = 0

while True:
    # print(l_pred, r_pred)
    success,f_img = cap.read()
    f_gray = cv2.cvtColor(f_img,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(f_gray, 1.1, 4)
    l_eye = l_eye_cascade.detectMultiScale(f_gray,1.1,4)
    r_eye = r_eye_cascade.detectMultiScale(f_gray,1.1,4)
    # f_crop = np.zeros((400,600))
    for (x, y, w, h) in face:

        cv2.rectangle(f_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #f_crop = f_img[y:y+h,x:x+w]

    for (x,y,w,h) in l_eye:
        f_l_eye = f_gray[y:y+h,x:x+w]
        f_l_eye_resize = cv2.resize(f_l_eye,(25,25))
        f_l_eye_resize = f_l_eye_resize/255
        f_l_eye_resize = f_l_eye_resize.reshape(-1,25,25,1)
        print(model.predict(f_l_eye_resize))
        l_pred = (model.predict(f_l_eye_resize) > 0.5).astype('int32')

    for (x,y,w,h) in r_eye:
        f_r_eye = f_gray[y:y+h,x:x+w]
        f_r_eye_resize = cv2.resize(f_r_eye,(25,25))
        f_r_eye_resize = f_r_eye_resize/255
        f_r_eye_resize = f_r_eye_resize.reshape(-1,25, 25,1)
        print(model.predict(f_r_eye_resize))
        r_pred = (model.predict(f_r_eye_resize) > 0.5).astype('int32')

    if l_pred == 1 and r_pred == 1:
        status = 'closed'
        sleep_count+=1

    elif l_pred == 1:
        status = 'closed'
        sleep_count += 0.25
    elif r_pred == 1:
        status = 'closed'
        sleep_count += 0.25

    else:
        status = 'open'
        sleep_count -= 1


    if sleep_count > 20:
        cv2.putText(f_img, f"Too Sleepy (SC) : {sleep_count} ---- Alert", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 0, 255), thickness=2)

    if status == 'closed':
        count += 1
        if count > 5:
            try:
                while True:
                    sound.play()
                    if cv2.waitKey(1) & 0xFF == ord('m'):
                        break
            except:  # isplaying = False
                pass
    else:
        count = 0


    cv2.imshow("Current Status",f_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
