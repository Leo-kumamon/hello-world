from ultralytics import YOLO

# 1. 模型路径（和你 val 用的一样）
model_path = r"C:\Users\DH\PythonProject1\runs\detect\people_car_yolo11n2\weights\best.pt"

# 2. 推理图片路径：直接用 val 目录
source_path = r"C:\Users\DH\PythonProject1\datasets\people-car\images\val"

# 3. 加载模型
model = YOLO(model_path)

# 4. 运行预测（全部参数用最常规的）
results = model.predict(
    source=source_path,
    imgsz=640,
    conf=0.25,          # 和 val 默认接近
    iou=0.7,
    save=True,
    save_conf=True,
    save_txt=False,
    project=r"C:\Users\DH\PythonProject1\runs\detect",
    name="debug_val3",  # 新目录名，避免和之前混
    exist_ok=True,
    verbose=True,
)

print("输出目录：", r"F:\data saved1")
