"""
YOLOv8n 车辆检测脚本
"""
import time
from pathlib import Path

import cv2
import numpy as np
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
        "imgsz": 1280,
        "iou": 0.5,
        "augment": False,
        "debug_print": False,
        "filter_vehicle_only": True,
        "use_preprocess": True,
    },
    "paths": {
        "input_dir": "data",
        "input_image": "",
        "output_dir": "results",
    },
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}


def preprocess_image(image_path: str, output_path: str) -> str:
    """图像预处理：去噪 + 对比度增强 + 轻锐化（兼容中文路径）"""
    image_bytes = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return image_path

    # 1) 轻度去噪（保边）
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

    # 2) CLAHE增强亮度通道对比度
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    # 3) 轻锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    ok, encoded = cv2.imencode(Path(output_path).suffix or ".jpg", sharpened)
    if not ok:
        return image_path
    encoded.tofile(output_path)
    return output_path

def load_config(config_path: str = "config/yolov8n.yaml") -> dict:
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
    """
    使用YOLOv8n检测车辆
    
    Args:
        image_path: 输入图像路径
    
    Returns:
        包含检测结果的字典
    """
    print(f"检测: {Path(image_path).name}")
    
    model_name = config["model"]["name"]
    weights = str(Path(config["model"]["weights"]))
    conf_threshold = float(config["detection"]["conf_threshold"])
    imgsz = int(config["detection"].get("imgsz", 640))
    iou = float(config["detection"].get("iou", 0.7))
    augment = bool(config["detection"].get("augment", False))
    debug_print = bool(config["detection"].get("debug_print", False))
    filter_vehicle_only = bool(config["detection"].get("filter_vehicle_only", True))
    use_preprocess = bool(config["detection"].get("use_preprocess", True))

    infer_image_path = image_path
    if use_preprocess:
        temp_preprocessed = str(Path("results") / "yolo" / f"preprocessed_{Path(image_path).stem}.jpg")
        Path(temp_preprocessed).parent.mkdir(parents=True, exist_ok=True)
        infer_image_path = preprocess_image(image_path, temp_preprocessed)

    # 加载模型
    model = YOLO(weights)
    
    # 记录推理时间
    start_time = time.time()
    results = model.predict(
        source=infer_image_path,
        conf=conf_threshold,
        iou=iou,
        imgsz=imgsz,
        augment=augment,
        verbose=False,
    )
    elapsed_ms = (time.time() - start_time) * 1000
    
    # 解析检测结果
    detections = []
    if results and len(results) > 0:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]

            if debug_print:
                print(f"class_name={class_name}, confidence={confidence:.3f}")

            if (not filter_vehicle_only) or (class_name.lower() in VEHICLE_CLASSES):
                detections.append({
                    'class_name': class_name,
                    'confidence': round(confidence, 3),
                    'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
    
    # 构建结果
    result = {
        'image': Path(image_path).name,
        'model': model_name,
        'model_params': '25.9M' if model_name.lower() == 'yolov8m' else '11.2M',
        'detections': detections,
        'num_detections': len(detections),
        'inference_time_ms': round(elapsed_ms, 1),
        'conf_threshold': conf_threshold,
        'iou': iou,
        'imgsz': imgsz,
        'augment': augment,
        'use_preprocess': use_preprocess,
        'filter_vehicle_only': filter_vehicle_only,
        'weights': weights,
    }
    
    return result


def main():
    """主函数"""
    config = load_config()
    input_dir = Path(config["paths"]["input_dir"])
    input_image = str(config["paths"].get("input_image", "")).strip()
    output_root = Path(config["paths"]["output_dir"])
    output_dir = output_root / "yolo"
    models_dir = Path("models")

    print("\n" + "="*60)
    print(f" {config['model']['name']} 车辆检测")
    print("="*60)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入目录
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        print("   请创建 data 文件夹并放入图片")
        return

    if input_image:
        image_path = Path(input_image)
        if not image_path.is_absolute():
            image_path = Path(input_image)
    else:
        image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg")))
        if not image_files:
            print(f"{input_dir} 文件夹中没有找到图片")
            return
        image_path = image_files[0]
        if len(image_files) > 1:
            print(f"找到 {len(image_files)} 张图片，默认使用第一张: {image_path.name}")

    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return

    try:
        result = detect_vehicles(str(image_path), config)
        print_summary(result)

        annotated_path = output_dir / f"yolov8n_{image_path.stem}.jpg"
        draw_boxes(str(image_path), result['detections'], str(annotated_path))
        if not result['detections']:
            print("   未检测到车辆")

        output_json = output_dir / "detections.json"
        save_json(result, str(output_json))

        print("\n" + "="*60)
        print(" 检测完成")
        print("="*60)
        print(f"图片: {image_path}")
        print(f"配置文件: config/yolov8n.yaml")
        print(f"结果位置:")
        print(f"   JSON: {output_json}")
        print(f"   标注图: {annotated_path}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
