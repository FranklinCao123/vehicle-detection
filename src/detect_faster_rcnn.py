"""
Faster R-CNN 车辆检测脚本
"""
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

from utils import save_json, print_summary


DEFAULT_CONFIG = {
    "model": {
        "name": "Faster R-CNN",
        "weights": "torchvision::fasterrcnn_resnet50_fpn",
        "device": "cpu",
    },
    "detection": {
        "conf_threshold": 0.3,
        "min_size": 800,
        "max_size": 1333,
    },
    "paths": {
        "input_dir": "data",
        "input_image": "",
        "output_dir": "results",
    },
}

# COCO车辆相关类别（内部过滤）
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}


def load_config(config_path: str = "config/faster_rcnn.yaml") -> dict:
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


def draw_boxes_pil(image_path: str, detections: list, output_path: str):
    """使用PIL绘制检测框"""
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        drawer.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        drawer.text((x1, max(0, y1 - 12)), f"vehicle {det['confidence']:.2f}", fill=(0, 255, 0))

    image.save(output_path)
    print(f"标注图已保存: {output_path}")


def detect_vehicles(image_path: str, config: dict) -> dict:
    """使用Faster R-CNN检测车辆"""
    print(f"检测: {Path(image_path).name}")

    conf_threshold = float(config["detection"]["conf_threshold"])
    min_size = int(config["detection"].get("min_size", 800))
    max_size = int(config["detection"].get("max_size", 1333))
    device = torch.device(config["model"].get("device", "cpu"))

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size).to(device)
    model.eval()

    preprocess = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(tensor)[0]
    elapsed_ms = (time.time() - start_time) * 1000

    categories = weights.meta.get("categories", [])
    detections = []

    for box, score, label in zip(output.get("boxes", []), output.get("scores", []), output.get("labels", [])):
        confidence = float(score)
        if confidence < conf_threshold:
            continue

        label_id = int(label)
        class_name = categories[label_id] if 0 <= label_id < len(categories) else "unknown"
        if class_name.lower() not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = box.tolist()
        detections.append(
            {
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )

    return {
        "image": Path(image_path).name,
        "model": config["model"]["name"],
        "model_params": "~41M",
        "detections": detections,
        "num_detections": len(detections),
        "inference_time_ms": round(elapsed_ms, 1),
        "conf_threshold": conf_threshold,
        "min_size": min_size,
        "max_size": max_size,
        "weights": config["model"]["weights"],
    }


def main():
    """主函数"""
    config = load_config()
    input_dir = Path(config["paths"]["input_dir"])
    input_image = str(config["paths"].get("input_image", "")).strip()
    output_dir = Path(config["paths"]["output_dir"]) / "faster_rcnn"

    print("\n" + "=" * 60)
    print(f" {config['model']['name']} 车辆检测")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

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

    result = detect_vehicles(str(image_path), config)
    print_summary(result)

    annotated_path = output_dir / f"faster_rcnn_{image_path.stem}.jpg"
    draw_boxes_pil(str(image_path), result["detections"], str(annotated_path))
    if not result["detections"]:
        print("   未检测到车辆")

    output_json = output_dir / "detections.json"
    save_json(result, str(output_json))

    print("\n" + "=" * 60)
    print(" 检测完成")
    print("=" * 60)
    print(f"图片: {image_path}")
    print(f"配置文件: config/faster_rcnn.yaml")
    print("结果位置:")
    print(f"   JSON: {output_json}")
    print(f"   标注图: {annotated_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
