"""
Faster R-CNN 车辆检测脚本 - 预处理版本（CLAHE + 锐化）
"""
import time
from pathlib import Path

import torch
import yaml
import cv2
import numpy as np
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
        "conf_threshold": 0.1,
        "min_size": 800,
        "max_size": 1333,
    },
    "preprocessing": {
        "enabled": True,
        "clahe_clip_limit": 2.5,
        "sharpen": True,
    },
    "paths": {
        "input_dir": "data",
        "input_image": "",
        "output_dir": "results",
    },
}

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


def read_image(image_path: Path):
    """读取图片，支持中文路径"""
    try:
        img_array = np.fromfile(str(image_path), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"读取失败: {e}")
        return None


def save_image(image, save_path: Path):
    """保存图片，支持中文路径"""
    cv2.imencode('.jpg', image)[1].tofile(str(save_path))


def preprocess_image(image: np.ndarray, clip_limit: float = 2.5, sharpen: bool = True) -> np.ndarray:
    """预处理：CLAHE对比度增强 + 可选锐化"""
    # 1. CLAHE增强（转LAB，只增强亮度通道）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 2. 锐化（增强边缘）
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    """将numpy数组转换为PIL Image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def draw_all_boxes_pil(image_path: str, detections: list, output_path: str):
    """使用PIL绘制所有检测框（不同类别不同颜色）"""
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)
    
    colors = {}
    color_list = [
        "#00FF00", "#FF0000", "#0000FF", "#FFFF00",
        "#FF00FF", "#00FFFF", "#FFA500", "#FF1493",
    ]
    
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det.get("class", "unknown")
        confidence = det["confidence"]
        
        if class_name not in colors:
            colors[class_name] = color_list[len(colors) % len(color_list)]
        color = colors[class_name]
        
        drawer.rectangle([x1, y1, x2, y2], outline=color, width=2)
        drawer.text((x1, max(0, y1 - 12)), f"{class_name}: {confidence:.2f}", fill=color)
    
    # 图例
    legend_y = 20
    for class_name, color in colors.items():
        drawer.rectangle([10, legend_y - 8, 30, legend_y + 2], fill=color)
        drawer.text((40, legend_y - 5), class_name, fill="white")
        legend_y += 20
    
    image.save(output_path)
    print(f"标注图（所有物体）: {output_path}")


def draw_boxes_pil(image_path: str, detections: list, output_path: str):
    """使用PIL绘制检测框（只绘制车辆）"""
    image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        drawer.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        drawer.text((x1, max(0, y1 - 12)), f"vehicle {det['confidence']:.2f}", fill=(0, 255, 0))

    image.save(output_path)
    print(f"车辆标注图: {output_path}")


def detect_vehicles(image_path: Path, config: dict) -> dict:
    """使用Faster R-CNN检测所有物体"""
    print(f"检测: {image_path.name}")
    
    pre_cfg = config.get("preprocessing", {})
    device = torch.device(config["model"].get("device", "cpu"))
    
    # 读取原图（支持中文路径）
    original_img = read_image(image_path)
    if original_img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 预处理
    if pre_cfg.get("enabled", True):
        clip_limit = pre_cfg.get("clahe_clip_limit", 2.5)
        sharpen = pre_cfg.get("sharpen", True)
        processed_img = preprocess_image(original_img, clip_limit, sharpen)
        
        pre_steps = [f"CLAHE(强度={clip_limit})"]
        if sharpen:
            pre_steps.append("锐化")
        print(f"   预处理: {' + '.join(pre_steps)}")
        
        # 保存预处理后的图片
        output_dir = Path(config["paths"]["output_dir"]) / "faster_rcnn"
        output_dir.mkdir(parents=True, exist_ok=True)
        preprocessed_path = output_dir / f"preprocessed_{image_path.stem}.jpg"
        save_image(processed_img, preprocessed_path)
        print(f"   预处理图片已保存: {preprocessed_path}")
        
        image = numpy_to_pil(processed_img)
    else:
        image = Image.open(image_path).convert("RGB")
        print(f"   预处理: 无")
    
    # 模型参数
    conf_threshold = float(config["detection"]["conf_threshold"])
    min_size = int(config["detection"].get("min_size", 800))
    max_size = int(config["detection"].get("max_size", 1333))
    
    # 加载模型
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, min_size=min_size, max_size=max_size).to(device)
    model.eval()
    
    # 预处理
    preprocess = weights.transforms()
    tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 检测
    start_time = time.time()
    with torch.no_grad():
        output = model(tensor)[0]
    elapsed_ms = (time.time() - start_time) * 1000
    
    # 解析结果（不限制类别）
    categories = weights.meta.get("categories", [])
    all_detections = []
    vehicle_detections = []
    
    print(f"\n   检测到的所有物体:")
    for box, score, label in zip(output.get("boxes", []), output.get("scores", []), output.get("labels", [])):
        confidence = float(score)
        if confidence < conf_threshold:
            continue
        
        label_id = int(label)
        class_name = categories[label_id] if 0 <= label_id < len(categories) else "unknown"
        
        x1, y1, x2, y2 = box.tolist()
        
        all_detections.append({
            "class": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
        
        print(f"     - {class_name}: conf={confidence:.3f}, "
              f"size={int(x2-x1)}x{int(y2-y1)}")
        
        if class_name.lower() in VEHICLE_CLASSES:
            vehicle_detections.append({
                "confidence": round(confidence, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
    
    return {
        "image": image_path.name,
        "model": config["model"]["name"],
        "model_params": "~41M",
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
    output_dir = Path(config["paths"]["output_dir"]) / "faster_rcnn"

    print("\n" + "=" * 60)
    print(f" {config['model']['name']} 车辆检测 - 预处理版")
    print("=" * 60)
    
    pre_cfg = config.get("preprocessing", {})
    if pre_cfg.get("enabled", True):
        print(f"预处理配置:")
        print(f"  CLAHE强度: {pre_cfg.get('clahe_clip_limit', 2.5)}")
        print(f"  锐化: {pre_cfg.get('sharpen', True)}")
    else:
        print("预处理: 禁用")
    print("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    if input_image:
        image_path = Path(input_image)
    else:
        images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
        if not images:
            print(f"{input_dir} 文件夹中没有找到图片")
            return
        image_path = images[0]

    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return

    result = detect_vehicles(image_path, config)
    print_summary(result)

    if result["all_detections"]:
        annotated_path = output_dir / f"faster_rcnn_all_{image_path.stem}.jpg"
        draw_all_boxes_pil(str(image_path), result["all_detections"], str(annotated_path))
        
        if result["detections"]:
            vehicle_annotated_path = output_dir / f"faster_rcnn_vehicle_{image_path.stem}.jpg"
            draw_boxes_pil(str(image_path), result["detections"], str(vehicle_annotated_path))
    else:
        print("未检测到任何物体")

    output_json = output_dir / "detections.json"
    save_json(result, str(output_json))

    print("\n" + "=" * 60)
    print(" 检测完成")
    print("=" * 60)
    print(f"图片: {image_path}")
    print(f"结果位置:")
    print(f"   JSON: {output_json}")
    if result["all_detections"]:
        print(f"   标注图: {annotated_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()