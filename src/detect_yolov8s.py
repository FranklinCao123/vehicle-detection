"""
YOLOv8s 车辆检测脚本
"""
import time
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import save_json, draw_boxes, print_summary


DEFAULT_CONFIG = {
    "model": {
        "name": "YOLOv8s",
        "weights": "models/yolov8s.pt",
    },
    "detection": {
        "conf_threshold": 0.1,
        "imgsz": 960,
    },
    "paths": {
        "input_dir": "data",
        "input_image": "",
        "output_dir": "results",
    },
}

# COCO车辆相关类别
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}


def load_config(config_path: str = "config/yolov8s.yaml") -> dict:
    """从配置文件读取参数"""
    config_file = Path(config_path)
    if not config_file.exists():
        return DEFAULT_CONFIG

    with open(config_file, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = DEFAULT_CONFIG.copy()
    for section_name, section_value in user_config.items():
        if isinstance(section_value, dict) and section_name in config:
            config[section_name] = {**config[section_name], **section_value}
        else:
            config[section_name] = section_value
    return config


def detect_vehicles(image_path: str, config: dict) -> dict:
    """使用YOLOv8s检测车辆"""
    print(f"检测: {Path(image_path).name}")

    model_name = config["model"]["name"]
    weights = str(Path(config["model"]["weights"]))
    conf_threshold = float(config["detection"]["conf_threshold"])
    imgsz = int(config["detection"].get("imgsz", 640))

    model = YOLO(weights)

    start_time = time.time()
    results = model.predict(source=image_path, conf=conf_threshold, imgsz=imgsz, verbose=False)
    elapsed_ms = (time.time() - start_time) * 1000

    detections = []
    if results and len(results) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if class_name.lower() in VEHICLE_CLASSES:
                detections.append(
                    {
                        "confidence": round(confidence, 3),
                        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                )

    result = {
        "image": Path(image_path).name,
        "model": model_name,
        "model_params": "11.2M",
        "detections": detections,
        "num_detections": len(detections),
        "inference_time_ms": round(elapsed_ms, 1),
        "conf_threshold": conf_threshold,
        "imgsz": imgsz,
        "weights": weights,
    }
    return result


def main():
    """主函数"""
    config = load_config()
    input_dir = Path(config["paths"]["input_dir"])
    input_image = str(config["paths"].get("input_image", "")).strip()
    output_root = Path(config["paths"]["output_dir"])
    output_dir = output_root / "yolov8s"
    models_dir = Path("models")

    print("\n" + "=" * 60)
    print(f" {config['model']['name']} 车辆检测")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    if input_image:
        image_path = Path(input_image)
    else:
        image_files = sorted(
            list(input_dir.glob("*.jpg"))
            + list(input_dir.glob("*.png"))
            + list(input_dir.glob("*.jpeg"))
        )
        if not image_files:
            print(f"{input_dir} 文件夹中没有找到图片")
            return
        image_path = image_files[0]

    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return

    try:
        result = detect_vehicles(str(image_path), config)
        print_summary(result)

        annotated_path = output_dir / f"yolov8s_{image_path.stem}.jpg"
        draw_boxes(str(image_path), result["detections"], str(annotated_path))
        if not result["detections"]:
            print("   未检测到车辆")

        output_json = output_dir / "detections.json"
        save_json(result, str(output_json))

        print("\n" + "=" * 60)
        print(" 检测完成")
        print("=" * 60)
        print(f"图片: {image_path}")
        print(f"配置文件: config/yolov8s.yaml")
        print("结果位置:")
        print(f"   JSON: {output_json}")
        print(f"   标注图: {annotated_path}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
