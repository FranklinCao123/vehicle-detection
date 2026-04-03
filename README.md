# 车辆检测技术测评报告

本项目使用本地代码完成车辆检测。

项目仓库：https://github.com/FranklinCao123/vehicle-detection.git

## 1) 使用的模型名称及版本

本次共测试 3 种模型：

1. YOLO8n
   - 脚本：src/detect_yolov8n.py
   - 权重：models/yolov8s.pt
   - 结果目录：results/yolo/

2. YOLOv8s
   - 脚本：src/detect_yolov8s.py
   - 权重：models/yolov8s.pt
   - 结果目录：results/yolov8s/

3. Faster R-CNN (ResNet50-FPN)
   - 脚本：src/detect_faster_rcnn.py
   - 权重：torchvision::fasterrcnn_resnet50_fpn
   - 结果目录：results/faster_rcnn/

## 2) 运行环境（框架、语言等）

- 操作系统：Windows
- 语言：Python 3
- 深度学习框架：PyTorch
- 检测框架：Ultralytics + torchvision
- 主要依赖：
  - torch>=2.0.0
  - torchvision>=0.15.0
  - ultralytics>=8.0.0
  - opencv-python>=4.8.0
  - Pillow>=10.0.0
  - pyyaml>=6.0

## 3) 检测到的目标位置（坐标或标注截图）

输入图片：待检测图片.png

### 三模型结果对比

| 模型 | 参数量 | 检测数 | 推理时间(ms) | 目标坐标(bbox) | 结果文件 |
|---|---:|---:|---:|---|---|
| YOLO（轻量配置） | 3.2M | 1 | 344.0 | [677.6, 233.5, 704.0, 248.0] | results/yolo/detections.json |
| YOLOv8s | 11.2M | 1 | 328.9 | [677.6, 233.5, 704.0, 248.0] | results/yolov8s/detections.json |
| Faster R-CNN | ~41M | 1 | 1664.9 | [677.9, 234.2, 703.2, 248.0] | results/faster_rcnn/detections.json |

### 标注截图路径

- YOLO（轻量配置）：results/yolo/
- YOLOv8s：results/yolov8s/
- Faster R-CNN：results/faster_rcnn/

## 4) 核心代码片段

以下为统一检测核心逻辑（以 YOLO 为例）：

```python
from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")
results = model.predict(
    source="data/待检测图片.png",
    conf=0.1,
    imgsz=1280,
    verbose=False,
)

detections = []
for box in results[0].boxes:
    cls_name = results[0].names[int(box.cls[0])]
    if cls_name.lower() in VEHICLE_CLASSES:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "confidence": float(box.conf[0]),
            "bbox": [x1, y1, x2, y2]
        })
```

## 运行方式

```bash
python src/detect_yolov8n.py
python src/detect_yolov8s.py
python src/detect_faster_rcnn.py
```
