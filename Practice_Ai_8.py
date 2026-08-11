import cv2
import numpy as np
import ultralytics

cap = cv2.VideoCapture("/Users/macbook/ITStep-AI/data/lesson_pose/sitting.mp4")
success, img = cap.read()

# cv2.imshow("img", img)

model = ultralytics.YOLO("yolo11s-pose.pt")

results = model.predict(img,device="cpu")
result = results[0]
print(result.boxes)

result_img = result.plot()

cv2.imshow("result", result_img)

keypoints = result.keypoints

xy = keypoints.xy
xy = xy.numpy().astype(int)

xy = xy[0]

x_left_knee, y_left_knee = xy[14]
x_left_arm, y_left_arm = xy[10]
x_right_arm, y_right_arm = xy[9]
x_right_knee, y_right_knee = xy[13]
cv2.circle(
    img,   # зображення де малювати коло
    center=(x_right_arm, y_right_arm),   # координати центру
    radius=15,   # радіус в пікселях
    color=(255, 0, 0),  # колір в BGR(синій)
    thickness=-1,)

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_left_arm, y_left_arm),   # координати центру
    radius=15,   # радіус в пікселях
    color=(0, 0, 255),  # колір в BGR(синій)
    thickness=-1,)

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_left_knee, y_left_knee),   # координати центру
    radius=15,   # радіус в пікселях
    color=(0, 255, 0),  # колір в BGR(синій)
    thickness=-1,)

cv2.circle(
    img,   # зображення де малювати коло
    center=(x_right_knee, y_right_knee),   # координати центру
    radius=15,   # радіус в пікселях
    color=(255, 255, 255),  # колір в BGR(синій)
    thickness=-1,)

cv2.imshow("result", img)

total_sitting = 0
is_sitting = True

# while True:
#     success,img = cap.read()
#     if not(success):
#         break
#
#     results = model.predict(img, device="cpu")
#     result = results[0]
#     result_img = result.plot()
#     cv2.imshow("result_plot", result_img)
#     keypoints = result.keypoints
#
#     xy = keypoints.xy
#     xy = xy.numpy().astype(int)
#
#     xy = xy[0]
#     x_left_knee, y_left_knee = xy[14]
#     x_left_arm, y_left_arm = xy[10]
#     x_right_arm, y_right_arm = xy[9]
#
#     cv2.circle(
#         img,  # зображення де малювати коло
#         center=(x_right_arm, y_right_arm),  # координати центру
#         radius=15,  # радіус в пікселях
#         color=(255, 0, 0),  # колір в BGR(синій)
#         thickness=-1, )
#
#     cv2.circle(
#         img,  # зображення де малювати коло
#         center=(x_left_arm, y_left_arm),  # координати центру
#         radius=15,  # радіус в пікселях
#         color=(0, 0, 255),  # колір в BGR(синій)
#         thickness=-1, )
#
#     cv2.circle(
#         img,  # зображення де малювати коло
#         center=(x_left_knee, y_left_knee),  # координати центру
#         radius=15,  # радіус в пікселях
#         color=(0, 255, 0),  # колір в BGR(синій)
#         thickness=-1, )
#
#     cv2.circle(
#         img,  # зображення де малювати коло
#         center=(x_right_knee, y_right_knee),  # координати центру
#         radius=15,  # радіус в пікселях
#         color=(255, 255, 255),  # колір в BGR(синій)
#         thickness=-1, )
#
#     if y_right_knee < y_left_arm and is_sitting:
#         total_sitting += 1
#
#     if y_right_knee < y_left_arm:
#         is_sitting = False
#     else:
#         is_sitting = True
#
#
#
#     cv2.putText(
#         img,  # зображення де пишемо текст
#         f"Total_Sitting:{total_sitting} Sitting:{is_sitting}",  # текст
#         (40,40),  # позиція, лівий нижній кут
#         cv2.FONT_HERSHEY_SIMPLEX,  # шрифт
#         1,  # розмір шрифту
#         (255, 255, 255),  # колір в BGR
#         2  # товщина ліній
#     )
#
#
#
#
#     cv2.imshow("result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


cv2.waitKey(0)