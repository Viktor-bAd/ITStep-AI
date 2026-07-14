import cv2


image = cv2.imread('sonet.png')

cv2.imshow('Original', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (3,3), 1.5)

res1 = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)

cv2.imshow('binar', res1)


image1 = cv2.imread('sonet_noised.png')

cv2.imshow('Original1', image1)

gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gauss1 = cv2.GaussianBlur(gray, (3,3), 1.5)

res2 = cv2.adaptiveThreshold(
    gauss1,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)

cv2.imshow('binar1', res2)

cv2.waitKey(0)