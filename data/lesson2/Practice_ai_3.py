# Відкрийте зображення data/lesson3/notes.png. Проведіть
# наступні дії:
#  проведіть бінарізацію(звичайну та адаптивну)
#  застосуйте розмиття(гаусове) візьміть ядра 3, 5, 11 та
# sigmaX 0, 2, 10
#  повторіть бінарізацію, але перед тим застосуйте bilateral
# filter
#
#
# import cv2
# image = cv2.imread('notes.png')
#
# cv2.imshow('Original', image)
# cv2.waitKey(0)
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # threshhold = 128
# #
# # mask = gray < threshhold
# #
# # gray[mask] = 0
# # gray[~mask] = 255
#
# bilat =  cv2.bilateralFilter(gray, 5, 75, 75)
# cv2.imshow('bilateral', bilat)
# cv2.waitKey(0)
#
# # gauss = cv2.GaussianBlur(gray, (3,3), 1.5)
# #
# # cv2.imshow('Gauss', gauss)
#
# res = cv2.adaptiveThreshold(
#     bilat,
#     255,
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     cv2.THRESH_BINARY,
#     11,
#     2
# )
#
# cv2.imshow('binar', res)
# cv2.waitKey(0)
import cv2

# Відкрийте зображення data/lesson3/sudoku.jpg. Проведіть
# для нього бінарізацію, а саме
#  CLAHE
#  гаусове розмиття
#  адаптивна бінарізація
#  NLMean


img = cv2.imread('sudoku.jpg')
cv2.imshow('Original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

result = clahe.apply(gray)
# cv2.imshow('CLAHE', result)

gauss = cv2.GaussianBlur(gray, (5,5), 5)
# cv2.imshow('Gauss', gauss)

res = cv2.adaptiveThreshold(
    result,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)
# cv2.imshow('CLAHE+Adaptive', res)

res1 = cv2.adaptiveThreshold(
    gauss,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)
cv2.imshow('Adeptive+Gauss', res1)

result_gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=3, searchWindowSize=21)
cv2.imshow('NLMean', result_gray)

res2 = cv2.adaptiveThreshold(
    result_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2,
)
cv2.imshow('Adeptive+NLMean', res2)

cv2.waitKey(0)
