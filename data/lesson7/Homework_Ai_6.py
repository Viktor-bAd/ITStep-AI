import cv2
import ultralytics

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture('/Users/macbook/ITStep-AI/data/lesson8/meetings.mp4')

people_threshold = 5
start_showing = False


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        device="mps",
        conf=0.25,
        iou=0.7,
        classes=[0],
        verbose=False
    )

    people_count = len(results[0].boxes)

    if not start_showing:
        if people_count >= people_threshold:
            start_showing = True
            print("Умову виконано! Починаємо показ відео.")
        else:
            continue

    result_frame = results[0].plot()
    display_frame = cv2.resize(result_frame, (960, 540))

    cv2.imshow('YOLO Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()