import cv2
import numpy as np

image = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

print(image.dtype)
print(image.shape)
print(image.max())
print(image.min())

cv2.imshow('Lenna Image', image)

cv2.waitKey(0)


segment = image[0:100, 0:50]
cv2.imshow('segment', segment)
cv2.waitKey(0)

segment2 = image[78:178, 78:178]
cv2.imshow('segment2', segment2)
print(segment2.shape)
cv2.waitKey(0)

segment4 = image[128:256, 0:256]
cv2.imshow('segment4', segment4)
cv2.waitKey(0)

segment5 = image[0:256, 0:128]
cv2.imshow('segment5', segment5)
cv2.waitKey(0)

segment6 = image[0:256, 128:256]
cv2.imshow('segment5', segment6)
cv2.waitKey(0)


image[0:20, 0:255] = 0
image[236:255, 0:255] = 255
cv2.imshow('image', image)
cv2.waitKey(0)

image[0:255, 0:20] = 0
image[0:255, 236:255] = 0
cv2.imshow('image', image)
cv2.waitKey(0)

image[0:40, 0:255] = 0
image[215:255, 0:255] = 0
image[0:255, 0:40] = 0
image[0:255, 215:255] = 0
cv2.imshow('image', image)
cv2.waitKey(0)


mask = image > 128
new_mask = mask.astype(np.uint8) * 255
cv2.imshow('mask', new_mask)
cv2.waitKey(0)

image[~mask] = 0
cv2.imshow('image', image)
cv2.waitKey(0)

