import numpy as np
import cv2

img_count = 0
cap = cv2.VideoCapture('game_screen.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    resize_img = cv2.resize(frame, (400, 300))
    cv2.imshow('frame', resize_img)
    key = cv2.waitKey(25) & 0xFF
    if key == ord('s'):
        path = 'img/' + str(img_count) + '.jpg'
        cv2.imwrite(path, resize_img)
        img_count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
