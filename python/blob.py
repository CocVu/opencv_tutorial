# Import relevant libraries
import numpy as np
import cv2

im1 = cv2.imread('/home/nam/Pictures/7LJ25.png')
# im1 = cv2.imread('/home/nam/Pictures/crop.png')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray scale', im1)
# Adaptive Threshold
thresh = cv2.adaptiveThreshold(im1, 255,
                            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                            thresholdType=cv2.THRESH_BINARY_INV,
                             # thresholdType=cv2.THRESH_BINARY,
                            blockSize=21,
                            C=2)

cv2.imshow('Threshold', thresh)

# # Morphology to close gaps
# se = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
# out = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

# # Find holes
# mask = np.zeros_like(im1)
# cv2.floodFill(out[1:-1,1:-1].copy(), mask, (0,0), 255)
# mask = (1 - mask).astype('bool')

# # Fill holes
# out[mask] = 255

# # Find contours
# # contours,_ = cv2.findContours(out.copy(),\
# #                               cv2.RETR_EXTERNAL,\
# #                               cv2.CHAIN_APPROX_SIMPLE)

# # (_, contours, _) = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# (_, contours, _) = cv2.findContours(out.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# # (im2, contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # Filter out contours with less than certain area
# area = 50
# filtered_contours = filter(lambda x: cv2.contourArea(x) > area,
#                            contours)

# # Draw final contours
# final = np.zeros_like(im1)
# cv2.drawContours(final, filtered_contours, -1, 255, -1)

# cv2.imshow('Shapes', final)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('train1_final.png', final)
