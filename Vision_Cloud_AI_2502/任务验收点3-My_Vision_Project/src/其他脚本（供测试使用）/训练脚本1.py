from ultralytics import YOLO

def main():
    # 1. 预训练权重路径，改成你自己的 yolo11n.pt 路径
    model = YOLO(r"F:\yolo11n.pt")

    # 2. 训练
    model.train(
        data=r"C:\Users\DH\PythonProject1\datasets\people-car\people-car.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device="cpu",       # 只用 CPU
        workers=0,
        name="people_car_yolo11n_cpu"
    )

if __name__ == "__main__":
    main()
