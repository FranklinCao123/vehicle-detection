"""
车辆区域分析工具 - 支持原图和预处理后图像对比
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime


def load_config(config_path: str = "config/analyze_config.yaml") -> dict:
    """读取分析配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def preprocess_image(image: np.ndarray, clip_limit: float = 2.0, sharpen: bool = False) -> np.ndarray:
    """预处理：CLAHE + 可选锐化"""
    # CLAHE增强
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 锐化
    if sharpen:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
    
    return result


def analyze_region(image, bbox, name, image_shape, output_dir: Path, prefix: str = ""):
    """分析单个区域，返回分析结果字典"""
    x1, y1, x2, y2 = bbox
    region = image[y1:y2, x1:x2]
    h, w = image_shape[:2]
    
    # 分析特征
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    edges = cv2.Canny(gray, 50, 150)
    
    # 计算指标
    area_ratio = (x2-x1)*(y2-y1)/(w*h)*100
    mean_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    highlight_ratio = np.sum(v_channel > 200) / v_channel.size * 100
    edge_ratio = np.sum(edges > 0) / edges.size * 100
    
    # 生成建议
    suggestions = []
    if brightness_std < 30:
        suggestions.append("对比度偏低，建议增强CLAHE(clip_limit=2.5-3.0)")
    if highlight_ratio < 5:
        suggestions.append("高亮区域少，建议用HSV增强亮度通道")
    if edge_ratio < 15:
        suggestions.append("边缘模糊，建议添加锐化")
    
    # 保存图片
    safe_name = name.replace(" ", "_")
    if prefix:
        region_filename = f"{prefix}_{safe_name}_region.jpg"
        marked_filename = f"{prefix}_{safe_name}_marked.jpg"
    else:
        region_filename = f"{safe_name}_region.jpg"
        marked_filename = f"{safe_name}_marked.jpg"
    
    region_path = output_dir / region_filename
    save_image(region, region_path)
    
    # 在原图上画框
    marked_image = image.copy()
    cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(marked_image, f"{prefix} {name}" if prefix else name, 
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    marked_path = output_dir / marked_filename
    save_image(marked_image, marked_path)
    
    return {
        "name": name,
        "prefix": prefix if prefix else "original",
        "bbox": [x1, y1, x2, y2],
        "size": [x2-x1, y2-y1],
        "area_ratio_percent": round(area_ratio, 2),
        "brightness": {
            "mean": round(mean_brightness, 1),
            "std": round(brightness_std, 1),
        },
        "highlight_ratio_percent": round(highlight_ratio, 1),
        "edge_ratio_percent": round(edge_ratio, 1),
        "suggestions": suggestions,
        "saved_images": {
            "region": str(region_path),
            "marked": str(marked_path),
        }
    }


def save_report(report_data: dict, output_dir: Path):
    """保存分析报告"""
    import json
    json_path = output_dir / "analysis_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    txt_path = output_dir / "analysis_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("车辆区域分析报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"图片: {report_data['image_path']}\n")
        f.write(f"图片尺寸: {report_data['image_width']} x {report_data['image_height']}\n")
        f.write(f"预处理参数: CLAHE clip_limit={report_data['preprocess_params']['clip_limit']}, "
                f"sharpen={report_data['preprocess_params']['sharpen']}\n\n")
        
        # 按区域分组显示原图和预处理后对比
        for region_name in report_data['region_names']:
            f.write("=" * 70 + "\n")
            f.write(f"区域: {region_name}\n")
            f.write("=" * 70 + "\n\n")
            
            # 原图数据
            orig = report_data['results']['original'][region_name]
            f.write("【原图】\n")
            f.write(f"  坐标: {orig['bbox']}\n")
            f.write(f"  尺寸: {orig['size'][0]} x {orig['size'][1]} 像素\n")
            f.write(f"  亮度标准差: {orig['brightness']['std']}\n")
            f.write(f"  边缘像素: {orig['edge_ratio_percent']}%\n")
            f.write(f"  高亮区域: {orig['highlight_ratio_percent']}%\n\n")
            
            # 预处理后数据
            proc = report_data['results']['preprocessed'][region_name]
            f.write("【预处理后】\n")
            f.write(f"  亮度标准差: {proc['brightness']['std']}\n")
            f.write(f"  边缘像素: {proc['edge_ratio_percent']}%\n")
            f.write(f"  高亮区域: {proc['highlight_ratio_percent']}%\n\n")
            
            # 改善幅度
            std_improve = proc['brightness']['std'] - orig['brightness']['std']
            edge_improve = proc['edge_ratio_percent'] - orig['edge_ratio_percent']
            f.write("【改善幅度】\n")
            f.write(f"  对比度变化: {std_improve:+.1f}\n")
            f.write(f"  边缘变化: {edge_improve:+.1f}%\n\n")
            
            if proc['suggestions']:
                f.write("【建议】\n")
                for suggestion in proc['suggestions']:
                    f.write(f"  - {suggestion}\n")
                f.write("\n")
        
        f.write("=" * 70 + "\n")
    
    return json_path, txt_path


def main():
    # 读取配置
    config = load_config()
    image_path_str = config["image_path"]
    image_path = Path(image_path_str)
    regions_config = config["regions"]
    
    # 预处理参数（从配置读取，或使用默认值）
    preprocess_config = config.get("preprocess", {"clip_limit": 2.5, "sharpen": True})
    clip_limit = preprocess_config.get("clip_limit", 2.5)
    sharpen = preprocess_config.get("sharpen", True)
    
    # 输出目录
    output_dir = Path("results") / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查文件
    if not image_path.exists():
        print(f"图片不存在: {image_path}")
        return
    
    # 读取原图
    img_original = read_image(image_path)
    if img_original is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 预处理
    img_preprocessed = preprocess_image(img_original, clip_limit, sharpen)
    
    h, w = img_original.shape[:2]
    print(f"图片: {image_path.name}")
    print(f"尺寸: {w} x {h}")
    print(f"预处理: CLAHE(clip_limit={clip_limit}), sharpen={sharpen}")
    print(f"输出目录: {output_dir}\n")
    
    # 分析原图
    print("分析原图...")
    original_results = {}
    for region in regions_config:
        name = region["name"]
        bbox = region["bbox"]
        result = analyze_region(img_original, bbox, name, img_original.shape, 
                                output_dir, prefix="original")
        original_results[name] = result
        print(f"  {name}: 亮度std={result['brightness']['std']}, 边缘={result['edge_ratio_percent']}%")
    
    # 分析预处理后
    print("\n分析预处理后...")
    preprocessed_results = {}
    for region in regions_config:
        name = region["name"]
        bbox = region["bbox"]
        result = analyze_region(img_preprocessed, bbox, name, img_preprocessed.shape,
                                output_dir, prefix="preprocessed")
        preprocessed_results[name] = result
        print(f"  {name}: 亮度std={result['brightness']['std']}, 边缘={result['edge_ratio_percent']}%")
    
    # 对比输出
    print("\n" + "=" * 50)
    print("对比结果")
    print("=" * 50)
    for region in regions_config:
        name = region["name"]
        orig = original_results[name]
        proc = preprocessed_results[name]
        
        print(f"\n{name}:")
        print(f"  对比度(亮度std): {orig['brightness']['std']:.1f} → {proc['brightness']['std']:.1f} "
              f"({proc['brightness']['std'] - orig['brightness']['std']:+.1f})")
        print(f"  边缘比例: {orig['edge_ratio_percent']:.1f}% → {proc['edge_ratio_percent']:.1f}% "
              f"({proc['edge_ratio_percent'] - orig['edge_ratio_percent']:+.1f}%)")
        print(f"  高亮比例: {orig['highlight_ratio_percent']:.1f}% → {proc['highlight_ratio_percent']:.1f}%")
    
    # 保存报告
    report_data = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_width": w,
        "image_height": h,
        "preprocess_params": {"clip_limit": clip_limit, "sharpen": sharpen},
        "region_names": [r["name"] for r in regions_config],
        "results": {
            "original": {name: original_results[name] for name in original_results},
            "preprocessed": {name: preprocessed_results[name] for name in preprocessed_results},
        }
    }
    
    json_path, txt_path = save_report(report_data, output_dir)
    
    print("\n" + "=" * 50)
    print("分析完成！")
    print("=" * 50)
    print(f"输出目录: {output_dir}")
    print(f"TXT报告: {txt_path}")
    print(f"JSON报告: {json_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()