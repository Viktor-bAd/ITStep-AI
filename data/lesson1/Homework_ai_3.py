import cv2
import numpy as np


img = cv2.imread('darken.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

v_eq = cv2.equalizeHist(v)
hsv_eq = cv2.merge([h, s, v_eq])
img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

v_float = v.astype(np.float32) * 1.3

v_bright = np.clip(v_float, 0, 255).astype(np.uint8)

hsv_bright = cv2.merge([h, s, v_bright])
img_bright = cv2.cvtColor(hsv_bright, cv2.COLOR_HSV2BGR)

cv2.imshow('Original', img)
cv2.imshow('Histogram', img_eq)
cv2.imshow('Brightness (30%)', img_bright)

cv2.waitKey(0)
