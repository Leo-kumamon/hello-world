
from ultralytics import YOLO
import os


model_path = r"C:\Users\DH\PythonProject1\runs\detect\people_car_yolo11n3（识别车很好用）\weights\best.pt"
model = YOLO(model_path)

source = r"F:\data for test"

# 3. 结果保存目录（可选，不填就默认 runs\detect\predict）
save_dir = r"F:\data saved3"

# 4. 运行推理
results = model.predict(
    source=source,
    imgsz=640,          # 输入图片尺寸，与训练时一致即可
    conf=0.01,          # 置信度阈值，调高会减少误检
    save=True,          # 保存带框的图片/视频
    project=save_dir,   # 保存到自定义目录
    name="",            # 为空则直接保存到 save_dir
    exist_ok=True       # 目录已存在也不报错
)

print("推理完成，结果已保存到：", os.path.abspath(save_dir))
