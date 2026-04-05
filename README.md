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
| **Faster R-CNN + 预处理** | ~41M | 2 | 1664.9 | [677.9, 234.2, 703.2, 248.0], [461.4, 228.4, 499.9, 245.2] | results/faster_rcnn/detections.json |

> **亮点：**
> - Faster R-CNN 在加入图像预处理（如CLAHE对比度增强、去噪、锐化）后，能够分辨出另一台原本YOLO系列无法检测到的车辆。

### 标注截图路径

- YOLO（轻量配置）：results/yolo/
- YOLOv8s：results/yolov8s/
- Faster R-CNN：results/faster_rcnn/

## 4) Faster R-CNN 检测核心代码片段

```python
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

def preprocess_image(image_path):
    # 预处理：CLAHE增强+锐化
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = preprocess_image('data/待检测图片.png')
transform = transforms.Compose([
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)[0]

# 只保留车辆类
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
    if label.item() in vehicle_classes and score.item() > 0.1:
        print(f"bbox: {box.tolist()}, conf: {score.item()}")
```

## 5) 结果分析与原理说明

- YOLO系列模型在小目标、远距离、低对比度场景下，特征表达能力有限，容易漏检。
- Faster R-CNN 结构更深，特征提取能力更强，结合图像预处理（如CLAHE对比度增强、锐化）后，能有效提升小目标和低对比度目标的可分辨性。
- 预处理提升了图像局部对比度和边缘信息，使得Faster R-CNN能检测出YOLO漏检的车辆。

## 运行方式

```bash
python src/detect_faster_rcnn.py
```

如需对比YOLO系列，运行：

```bash
python src/detect_yolov8n.py
python src/detect_yolov8s.py
```
