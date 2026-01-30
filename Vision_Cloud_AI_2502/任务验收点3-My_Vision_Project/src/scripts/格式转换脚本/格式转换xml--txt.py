import os
import xml.etree.ElementTree as ET


# VOC 标注 XML 所在目录
voc_anno_dir = r"C:\Users\DH\PythonProject1\datasets\people-car\annotations-xml\val"   # 这里要将val和train的分别导入
# 对应图片所在目录（只用于检查，不一定必须）
img_dir      = r"C:\Users\DH\PythonProject1\datasets\people-car\images\val"
# 转换后 YOLO txt 要保存的目录
yolo_label_dir = r"C:\Users\DH\PythonProject1\datasets\people-car\labels\val"


classes = [
    "people",
    "car",

]



os.makedirs(yolo_label_dir, exist_ok=True)

def convert_bbox(size, box):
    """
    VOC -> YOLO
    size: (w, h)
    box:  (xmin, ymin, xmax, ymax)
    return: (x_center, y_center, w, h) 归一化到 0~1
    """
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    # 防止越界
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    box_w = xmax - xmin
    box_h = ymax - ymin
    x_c = xmin + box_w / 2.0
    y_c = ymin + box_h / 2.0

    # 归一化
    return (
        x_c / img_w,
        y_c / img_h,
        box_w / img_w,
        box_h / img_h
    )

def convert_xml_one(xml_path, out_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 图像宽高
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    lines = []

    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in classes:
            # 如果遇到不在 classes 列表里的类别，直接跳过
            # 也可以改成 raise 报错
            continue
        cls_id = classes.index(cls_name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_c, y_c, w, h = convert_bbox((img_w, img_h), (xmin, ymin, xmax, ymax))
        line = f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
        lines.append(line)

    # 如果某张图没有任何目标，可以选择写一个空文件
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def batch_convert(voc_anno_dir, yolo_label_dir):
    xml_files = [f for f in os.listdir(voc_anno_dir) if f.endswith(".xml")]
    print("共发现 XML 标注文件:", len(xml_files))

    for i, xml_name in enumerate(xml_files, 1):
        xml_path = os.path.join(voc_anno_dir, xml_name)
        txt_name = os.path.splitext(xml_name)[0] + ".txt"
        out_txt_path = os.path.join(yolo_label_dir, txt_name)

        convert_xml_one(xml_path, out_txt_path)

        if i % 50 == 0 or i == len(xml_files):
            print(f"[{i}/{len(xml_files)}] 已处理: {xml_name}")

    print("转换完成，YOLO 标签保存在:", yolo_label_dir)


if __name__ == "__main__":
    batch_convert(voc_anno_dir, yolo_label_dir)
