import cv2
from ultralytics import YOLO
import utils


model = YOLO('yolo11s-pose.pt')

cap = cv2.VideoCapture("/Users/macbook/ITStep-AI/data/lesson_pose/squat.mp4")

count = 0
stage = None
LOWER_LIMIT = 90
UPPER_LIMIT = 160

while True:
    success, img = cap.read()

    if not success:
        break

    results = model(img, verbose=False)

    if results[0].keypoints is not None and len(results[0].keypoints.data[0]) > 0:
        kpts = results[0].keypoints.data[0].cpu().numpy()

        hip = kpts[11]
        knee = kpts[13]
        ankle = kpts[15]

        angle = utils.get_angle(hip[0], hip[1], knee[0], knee[1], ankle[0], ankle[1])

        if angle > UPPER_LIMIT:
            stage = "up"
        if angle < LOWER_LIMIT and stage == 'up':
            stage = "down"
            count += 1

        cv2.circle(
            img=img,
            center=(int(hip[0]), int(hip[1])),
            radius=15,
            color=(255, 255, 255),
            thickness=-1
        )

        cv2.circle(
            img=img,
            center=(int(knee[0]), int(knee[1])),
            radius=15,
            color=(255, 255, 255),
            thickness=-1
        )

        cv2.circle(
            img=img,
            center=(int(ankle[0]), int(ankle[1])),
            radius=15,
            color=(255, 255, 255),
            thickness=-1
        )

        cv2.putText(img, f"Angle: {int(angle)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(img, f"Squats: {count}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLO11s Squat Counter', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()