# Відкрийте зображення data/lesson2/marbles.png.
# Використайте кольорову сегментацію для отримання масок до
# кульок:
#  синього кольору
#  зеленого і червоного
#  чорного
#  білого
#  усіх кульок


import cv2

image = cv2.imread('/Users/macbook/ITStep-AI/data//lesson2/marbles.png')
cv2.imshow('image', image)
cv2.waitKey(0)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower = (100,100,100)
# upper = (130,255,255)
#
# mask_blue = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow('mask', mask_blue)
# cv2.waitKey(0)
#
#
# lower = (0,100,150)
# upper = (7,255,255)
#
# mask_red = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow('mask', mask_red)
# cv2.waitKey(0)
#
#
# lower = (40,90,80)
# upper = (85,255,255)
#
# mask_green = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow('mask', mask_green)
# cv2.waitKey(0)
#
#
# mask_both = cv2.bitwise_or(mask_red, mask_green)
#
# cv2.imshow('mask', mask_both)
# cv2.waitKey(0)


# lower = (0,0,0)
# upper = (100,100,40)
#
# mask_black = cv2.inRange(hsv, lower, upper)
#
# cv2.imshow('mask', mask_black)
# cv2.waitKey(0)

lower = (0,0,200)
upper = (180,30,255)

mask_white = cv2.inRange(hsv, lower, upper)

cv2.imshow('mask', mask_white)
cv2.waitKey(0)

