from ultralytics import YOLO
import cv2


model = YOLO("/Users/macbook/ITStep-AI/data/lesson_seg/brain-tumor-seg.pt")

image = cv2.imread("/Users/macbook/ITStep-AI/data/lesson_seg/tumor1.jpg")

cv2.imshow("image", image)

results = model(image)

mask = results[0].masks.data[0].cpu().numpy()

mask = (mask * 255).astype("uint8")

mask = cv2.resize(
    mask,
    (image.shape[1], image.shape[0])
)

area_pixels = cv2.countNonZero(mask)

area = area_pixels * 0.0025

if area < 10:
    tumor_type = "small"
elif area <= 25:
    tumor_type = "middle"
else:
    tumor_type = "large"

print("Площа в пікселях:", area_pixels)
print("Площа:", area)
print("Тип пухлини:", tumor_type)

tumor = cv2.bitwise_and(
    image,
    image,
    mask=mask
)

cv2.imwrite(
    f"data/lesson_seg/{tumor_type}.jpg",
    tumor
)

cv2.imshow(tumor_type, tumor)
cv2.waitKey(0)
cv2.destroyAllWindows()