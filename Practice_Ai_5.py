import cv2
import ultralytics
from torch.xpu import device

model = ultralytics.YOLO("yolo11s.pt")

cap = cv2.VideoCapture("/Users/macbook/ITStep-AI/data/lesson3/animals.mp4")

success, img = cap.read()

cv2.imshow("img",img)

results = model.predict(
    img,
    device="mps",
    conf=0.25,
    iou=0.7
)

result = results[0]

res = result.plot()

cv2.imshow("result",res)

boxes = result.boxes

box1 = boxes[0]

# print(box1)

cls = box1.cls
print(cls)

conf = box1.conf
print(conf)

xyxy = box1.xyxy
# print(xyxy)

cls = cls.cpu().numpy()
conf = conf.cpu().numpy()
xyxy = xyxy.cpu().numpy().astype(int)

x1, y1, x2, y2 = xyxy[0]

names = result.names
name1 = names[cls[0]]



box1_img = img[y1:y2, x1:x2]
cv2.imshow(f"img1{name1},{conf[0]*100:2f}",box1_img)


for box in boxes:
    cls = box.cls
    print(cls)

    conf = box.conf
    print(conf)

    xyxy = box.xyxy
    # print(xyxy)

    cls = cls.cpu().numpy()
    conf = conf.cpu().numpy()
    xyxy = xyxy.cpu().numpy().astype(int)

    x1, y1, x2, y2 = xyxy[0]

    names = result.names
    name1 = names[cls[0]]

    box1_img = img[y1:y2, x1:x2]
    cv2.imshow(f"img1{name1},{conf[0] * 100:2f}", box1_img)


while True:
    success, img = cap.read()
    if not success:
        break

    results = model.predict(
        img,
        device="mps",
        conf=0.25,
        iou=0.7
    )
    result = results[0]

    boxes = result.boxes

    for i in range((len(boxes))):
        box = boxes[i]
        cls = box.cls
        print(cls)

        conf = box.conf
        print(conf)

        xyxy = box.xyxy
        # print(xyxy)

        cls = cls.cpu().numpy()
        conf = conf.cpu().numpy()
        xyxy = xyxy.cpu().numpy().astype(int)

        x1, y1, x2, y2 = xyxy[0]

        names = result.names
        name = names[cls[0]]

        box1_img = img[y1:y2, x1:x2]
        cv2.imshow(f"img1{name}_{i}", box1_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

