from ultralytics import YOLO

# 1. 直接加载本地的 yolo11n.pt（不要再让它去网上下）
model = YOLO(r"F:\yolo11n.pt")

# 2. 预测一张图片
img_path = r"F:\R-C.jpg"   # 注意：不要有空格
results = model(img_path, save=True)

print("推理完成，结果保存在 runs/detect/predict 目录下")
