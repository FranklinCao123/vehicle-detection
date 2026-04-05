"""
YOLOv8s 车辆检测脚本 - 预处理版本（CLAHE + 锐化）
"""

import time
from pathlib import Path

import yaml
import cv2
import numpy as np
from ultralytics import YOLO

from utils import save_json, print_summary


DEFAULT_CONFIG = {
    "model": {
        "name": "YOLOv8s",
        "weights": "models/yolov8s.pt",
    },
    "detection": {
        "conf_threshold": 0.1,
        "imgsz": 640,
    },
    "preprocessing": {
        "enabled": True,
        "clahe_clip_limit": 2.0,
        "sharpen": True,
    },
    "paths": {
        "input_dir": "data",
        "input_image": "",
        "output_dir": "results",
    },
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}


def load_config(config_path: str = "config/yolov8s_preprocess.yaml") -> dict:
    """读取配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        return DEFAULT_CONFIG
    
    with open(config_file, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}
    
    config = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    return config


def simple_preprocess(image_path: str, clip_limit: float = 2.0, sharpen: bool = False) -> np.ndarray:
    """预处理：CLAHE对比度增强 + 可选锐化"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result


def draw_all_boxes(image_path: str, detections: list, output_path: str):
    """在原图上绘制所有检测到的物体（不同类别用不同颜色）"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # 为不同类别分配不同颜色
    colors = {}
    color_list = [
        (0, 255, 0),    # 绿色 - car
        (255, 0, 0),    # 蓝色 - truck
        (0, 0, 255),    # 红色 - bus
        (255, 255, 0),  # 青色 - motorcycle
        (255, 0, 255),  # 紫色 - bicycle
        (0, 255, 255),  # 黄色 - train
        (128, 0, 128),  # 紫色 - 其他
    ]
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = [int(coord) for coord in det["bbox"]]
        confidence = det["confidence"]
        class_name = det.get("class", "unknown")
        
        # 确保坐标在图片范围内
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 为每个类别分配颜色
        if class_name not in colors:
            colors[class_name] = color_list[len(colors) % len(color_list)]
        color = colors[class_name]
        
        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 添加标签（类别名 + 置信度）
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 添加图例
    legend_y = 30
    for class_name, color in colors.items():
        cv2.rectangle(img, (10, legend_y - 15), (30, legend_y), color, -1)
        cv2.putText(img, class_name, (40, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25
    
    cv2.imwrite(output_path, img)
    print(f"标注图（所有物体）: {output_path}")


def detect_vehicles(image_path: str, config: dict) -> dict:
    """检测所有物体（不限于车辆）"""
    print(f"检测: {Path(image_path).name}")
    
    pre_cfg = config.get("preprocessing", {})
    
    if pre_cfg.get("enabled", True):
        clip_limit = pre_cfg.get("clahe_clip_limit", 2.0)
        sharpen = pre_cfg.get("sharpen", False)
        processed_img = simple_preprocess(image_path, clip_limit, sharpen)
        source = processed_img
        
        pre_steps = [f"CLAHE(强度={clip_limit})"]
        if sharpen:
            pre_steps.append("锐化")
        print(f"   预处理: {' + '.join(pre_steps)}")
    else:
        source = image_path
        print(f"   预处理: 无")
    
    weights = str(Path(config["model"]["weights"]))
    conf_threshold = config["detection"]["conf_threshold"]
    imgsz = config["detection"]["imgsz"]
    
    model = YOLO(weights)
    
    start_time = time.time()
    results = model.predict(source=source, conf=conf_threshold, imgsz=imgsz, verbose=False)
    elapsed_ms = (time.time() - start_time) * 1000
    
    # 解析所有检测结果
    all_detections = []
    vehicle_detections = []
    
    if results and len(results) > 0:
        print(f"\n   检测到的所有物体:")
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            all_detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
            
            print(f"     - {class_name}: conf={confidence:.3f}, "
                  f"size={int(x2-x1)}x{int(y2-y1)}, "
                  f"位置=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
            
            if class_name.lower() in VEHICLE_CLASSES:
                vehicle_detections.append({
                    "confidence": round(confidence, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "class": class_name,
                })
    
    return {
        "image": Path(image_path).name,
        "model": config["model"]["name"],
        "all_detections": all_detections,
        "num_all_detections": len(all_detections),
        "detections": vehicle_detections,
        "num_detections": len(vehicle_detections),
        "inference_time_ms": round(elapsed_ms, 1),
        "conf_threshold": conf_threshold,
        "preprocessing_enabled": pre_cfg.get("enabled", True),
    }


def main():
    config = load_config()
    input_dir = Path(config["paths"]["input_dir"])
    input_image = config["paths"].get("input_image", "")
    output_dir = Path(config["paths"]["output_dir"]) / "preprocess_result"
    
    print("\n" + "=" * 50)
    print(" YOLOv8s 检测 - CLAHE + 锐化版")
    print("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_image:
        image_path = Path(input_image)
    else:
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
        if not images:
            print(f"没有找到图片: {input_dir}")
            return
        image_path = images[0]
    
    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return
    
    result = detect_vehicles(str(image_path), config)
    print_summary(result)
    
    # ===== 关键修改：标注所有检测到的物体 =====
    if result["all_detections"]:
        # 标注所有物体（不限于车辆）
        annotated_path = output_dir / f"result_all_{image_path.stem}.jpg"
        draw_all_boxes(str(image_path), result["all_detections"], str(annotated_path))
        
        # 如果只想要车辆标注，可以再保存一份
        if result["detections"]:
            vehicle_annotated_path = output_dir / f"result_vehicle_{image_path.stem}.jpg"
            from utils import draw_boxes
            draw_boxes(str(image_path), result["detections"], str(vehicle_annotated_path))
            print(f"车辆标注图: {vehicle_annotated_path}")
    else:
        print("未检测到任何物体")
    
    output_json = output_dir / "detections.json"
    save_json(result, str(output_json))
    print(f"结果JSON: {output_json}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()