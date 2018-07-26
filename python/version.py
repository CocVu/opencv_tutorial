import numpy as np
import imutils
import time
import cv2

print cv2.__version__
# cap = cv2.VideoCapture("~/Videos/TruongKimDong2.mp4")

# smooth = 25
# count = 0
# ret, frame1= cap.read()
# ret, frame2= cap.read()

# gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# gray1 = cv2.GaussianBlur(frame1, (smooth, smooth), 0)
# # smooth = cv2.bilateralFilter(frame1,9,75,75)
# gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.GaussianBlur(frame2, (smooth, smooth), 0)


# frameDelta = cv2.absdiff(gray2, gray1)
# thresh = cv2.threshold(frameDelta, 25, 50, cv2.THRESH_BINARY)[1]

# thresh = cv2.dilate(thresh, None, iterations=5)
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# cv2.imshow('frame',frameDelta)
# cv2.waitKey()
# cap.release()
# cv2.destroyAllWindows()

# cv2.imshow('Thresh', thresh)
# cv2.waitKey()
# cap.release()
# cv2.destroyAllWindows()
