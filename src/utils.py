"""
工具函数库
"""
import json
import cv2
import time
from pathlib import Path
from typing import List, Dict, Tuple


def save_json(data: Dict, path: str):
    """保存为JSON文件"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已保存: {path}")


def load_json(path: str) -> Dict:
    """读取JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_boxes(image_path: str, detections: List[Dict], output_path: str):
    """在图像上绘制检测框"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        conf = det['confidence']
        label = f"vehicle {conf:.2f}"
        
        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制文本
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(output_path, img)
    print(f"✓ 标注图已保存: {output_path}")


def print_summary(results: Dict):
    """打印结果摘要"""
    print("\n" + "="*60)
    print(f"检测结果")
    print("="*60)
    print(f"模型: {results.get('model')}")
    print(f"图像: {results.get('image')}")
    print(f"检测到: {len(results.get('detections', []))} 个车辆")
    print(f"推理时间: {results.get('inference_time_ms'):.1f}ms")
    print("-"*60)
    
    for i, det in enumerate(results['detections'], 1):
        x1, y1, x2, y2 = det['bbox']
        print(f"目标{i}: vehicle 置信度:{det['confidence']:.3f}  "
              f"位置:[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    print("="*60 + "\n")
