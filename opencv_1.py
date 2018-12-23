# import numpy as np
# import cv2
# img = cv2.imread("Glitch-d884f496-2438-48a5-b198-d26136fc9aa0.jpg",1)
# #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
# #cv2.imshow("Image",img)
# #cv2.waitKey(0)
# print(img.size)
# v=cv2.VideoCapture(0)
# while True:
#     ret, frame=v.read()
#     cv2.imshow('f',frame)
#     if cv2.waitKey(500) & 0xFF==ord('q'):
#         break
#
# v.release()




# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
#
# while (1):
#
#     _, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([30, 150, 50])
#     upper_red = np.array([255, 255, 180])
#
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#
#     cv2.imshow('Original', frame)
#     edges = cv2.Canny(frame, 50, 50)
#     cv2.imshow('Edges', edges)
#
#     k = cv2.waitKey(5) & 0xFF
#     if k == 27:
#         break
#
# cv2.destroyAllWindows()
# cap.release()


import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()