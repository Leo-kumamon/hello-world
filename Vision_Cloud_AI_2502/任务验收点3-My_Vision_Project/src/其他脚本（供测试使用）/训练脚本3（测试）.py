from ultralytics import YOLO


def main():
    # 1. 预训练权重路径：把下面改成你的 yolo11n.pt 实际路径
    # 例如在 F 盘：F:\yolo11n.pt
    model = YOLO(r"F:\yolo11n.pt")

    # 2. 数据集配置 yaml 路径
    data_yaml = r"F:\people-car\people-car.yaml"

    # 3. 开始训练
    model.train(
        data=data_yaml,
        epochs=100,                 # 训练轮数，可调整
        imgsz=640,                 # 输入图片尺寸
        batch=4,                   # 批大小，CPU 建议小一点
        device="cpu",              # 只用 CPU
        workers=0,                 # Windows + CPU 建议 0
        name="people_car_yolo11n"  # 结果目录 runs/detect/people_car_yolo11n
    )


if __name__ == "__main__":
    main()
