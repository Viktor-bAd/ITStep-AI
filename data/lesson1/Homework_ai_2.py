
import cv2
import numpy as np


img = cv2.imread('Lenna.png')
mask1_raw = cv2.imread('mask1.png', cv2.IMREAD_GRAYSCALE)
mask2_raw = cv2.imread('mask2.png', cv2.IMREAD_GRAYSCALE)

mask1_bool = mask1_raw.astype(bool)
mask2_bool = mask2_raw.astype(bool)

mask1 = mask1_bool.astype(np.uint8) * 255
mask2 = mask2_bool.astype(np.uint8) * 255

intersection_mask = cv2.bitwise_and(mask1, mask2)

res_mask1 = cv2.bitwise_and(img, img, mask=mask1)
res_mask2 = cv2.bitwise_and(img, img, mask=mask2)
res_intersection = cv2.bitwise_and(img, img, mask=intersection_mask)

cv2.imshow('mask1', res_mask1)
cv2.imshow('mask2', res_mask2)
cv2.imshow('mask1 AND mask2', res_intersection)

cv2.waitKey(0)


img = cv2.imread('baboo.jpg')

roi = img[20:60, 60:180]

cv2.imshow('ROI', roi)
cv2.waitKey(0)