import numpy as np
import cv2
import sys
# from hog import image_hog

font = cv2.FONT_HERSHEY_SIMPLEX
video_path = '/home/nam/Videos/test.mp4'

cv2.ocl.setUseOpenCL(False)

cap = cv2.VideoCapture(video_path)

fgbg = cv2.createBackgroundSubtractorMOG2()

while (cap.isOpened):
    n_contour = 0
    ret, frame = cap.read()

    if ret==True:

        fgmask = fgbg.apply(frame)

        (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #looping for contours
        print(hierarchy.shape)
        # print(len(contours))
        for c in contours:
            n = cv2.contourArea(c)
            if n < 2000:
                continue

            n_contour+=1

            (x,y,h,w) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            r_2,thresh1 = cv2.threshold(frame,127,255,cv2.THRESH_BINARY)

        cv2.imshow('foreground and background',fgmask)
        # cv2.imshow('foreground and background',thresh1)
        cv2.imshow('rgb',frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("get_filter",n_contour)
cap.release()
cv2.destroyAllWindows()
