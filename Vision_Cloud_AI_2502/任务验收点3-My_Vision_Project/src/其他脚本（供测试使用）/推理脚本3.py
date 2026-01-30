from ultralytics import YOLO
import cv2, os

img_path   = r"C:\Users\DH\PythonProject1\datasets\people-car\images\val\屏幕截图 2026-01-28 212331.png"
label_path = r"C:\Users\DH\PythonProject1\datasets\people-car\labels\val\屏幕截图 2026-01-28 212331.txt"
img = cv2.imread(img_path)
h, w = img.shape[:2]

# 画标注框（绿色）
if os.path.exists(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.strip().split())
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            x1 = int(cx - bw / 2)
            y1 = int(cy - bh / 2)
            x2 = int(cx + bw / 2)
            y2 = int(cy + bh / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

out_path = r"F:\data saved1"
cv2.imwrite(out_path, img)
print("保存到:", out_path)
