import numpy as np
import cv2




# Read in image and convert to grayscale
img = cv2.imread('/home/nam/Pictures/7LJ25.png')
# gray = cvtColor(im1, bwsrc, cv::COLOR_RGB2GRAY);
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

cv2.imshow('Shapes', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('train1_final.png', final)
