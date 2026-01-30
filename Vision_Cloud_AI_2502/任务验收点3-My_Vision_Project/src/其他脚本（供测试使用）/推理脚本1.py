from ultralytics import YOLO
import os

# 1. 加载模型
model_path = r"C:\Users\DH\PythonProject1\runs\detect\people_car_yolo11n2\weights\best.pt"
print("模型路径:", model_path)
print("模型文件是否存在:", os.path.exists(model_path))

model = YOLO(model_path)
print("模型 names:", model.names)   # 看看类别名有没有正常加载

# 2. 推理数据
source = r"C:\Users\DH\PythonProject1\datasets\people-car\images\val"
print("推理输入路径:", source)
print("输入路径是否存在:", os.path.exists(source))

save_dir = r"F:\data saved"

# 3. 运行推理
results = model.predict(
    source=source,
    imgsz=640,
    conf=0.01,
    save=True,
    project=save_dir,
    name="",
    exist_ok=True,
    verbose=True,   # 多打印一点信息
)

print("推理完成，结果已保存到：", os.path.abspath(save_dir))

# 4. 打印第1张图的检测结果信息（很关键）
if len(results):
    r0 = results[0]
    print("第1张图片检测到的框数量:", len(r0.boxes))
    if len(r0.boxes):
        print(r0.boxes.xyxy[:5])   # 前几个框的坐标
        print(r0.boxes.cls[:5])    # 对应的类别索引
        print(r0.boxes.conf[:5])   # 对应置信度
