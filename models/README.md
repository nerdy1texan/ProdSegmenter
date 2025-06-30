# Models Directory

This directory stores all model weights, checkpoints, and related artifacts for the ProdSegmenter project.

## 📁 Directory Structure

```
models/
├── sam/                        # Segment Anything Model weights
│   ├── sam_vit_h_4b8939.pth   # SAM ViT-Huge checkpoint
│   ├── sam_vit_l_0b3195.pth   # SAM ViT-Large checkpoint  
│   └── sam_vit_b_01ec64.pth   # SAM ViT-Base checkpoint
└── custom_segmentation/        # Trained segmentation models
    ├── deeplabv3plus/          # DeepLabV3+ model variants
    ├── unet/                   # UNet model variants
    ├── yolov8_seg/            # YOLOv8 segmentation models
    └── production/            # Production-ready models
```

## 🎯 Model Categories

### SAM Models (`sam/`)
**Purpose**: Pre-trained Segment Anything Models for bootstrapping
- **ViT-Huge**: Highest accuracy, slowest inference (6GB+ VRAM)
- **ViT-Large**: Balanced accuracy/speed (4GB+ VRAM)  
- **ViT-Base**: Fastest inference, good accuracy (2GB+ VRAM)

**Usage**: Initial mask generation and data labeling

### Custom Segmentation Models (`custom_segmentation/`)
**Purpose**: Models trained specifically for grocery shelf segmentation

#### DeepLabV3+ (`deeplabv3plus/`)
- Semantic segmentation with atrous convolution
- ResNet/MobileNet backbones
- Best for detailed segmentation boundaries

#### UNet (`unet/`)
- Encoder-decoder architecture
- Excellent for medical/precise segmentation
- Good balance of speed and accuracy

#### YOLOv8 Segmentation (`yolov8_seg/`)
- Real-time instance segmentation
- Optimized for speed (30-60 FPS)
- Suitable for product detection + segmentation

## 📋 Model Specifications

### File Naming Convention
```
{architecture}_{backbone}_{dataset}_{version}_{timestamp}.pth

Examples:
deeplabv3plus_resnet50_groceryshelf_v1_20241201.pth
unet_resnet34_groceryshelf_v2_20241205.pth
yolov8n_seg_groceryshelf_v1_20241210.pth
```

### Model Metadata
Each model includes companion files:
- `{model_name}_config.yaml`: Training configuration
- `{model_name}_metrics.json`: Performance metrics
- `{model_name}_classes.json`: Class definitions
- `{model_name}_preprocessing.json`: Input preprocessing specs

## 🚀 Model Usage

### Loading Models

```python
import torch
from pathlib import Path

# Load a trained segmentation model
model_path = Path("models/custom_segmentation/deeplabv3plus/best_model.pth")
model = torch.load(model_path, map_location='cpu')

# Load with configuration
import yaml
config_path = model_path.with_suffix('.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)
```

### Model Inference

```python
# Setup model for inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Process image
with torch.no_grad():
    outputs = model(input_tensor)
    predictions = torch.argmax(outputs, dim=1)
```

## 📊 Performance Benchmarks

### Real-time Requirements
Target performance for production deployment:

| Model | Input Size | FPS | Latency (ms) | mIoU | GPU Memory |
|-------|------------|-----|--------------|------|------------|
| YOLOv8n-seg | 640x640 | 60+ | <17 | 0.75+ | 2GB |
| YOLOv8s-seg | 640x640 | 45+ | <22 | 0.78+ | 3GB |
| UNet-ResNet34 | 512x512 | 35+ | <28 | 0.80+ | 4GB |
| DeepLabV3+-ResNet50 | 512x512 | 30+ | <33 | 0.82+ | 6GB |

### Accuracy Metrics
- **mIoU (mean Intersection over Union)**: Primary segmentation metric
- **Pixel Accuracy**: Overall classification accuracy
- **Class-wise IoU**: Per-category performance
- **Inference Time**: End-to-end processing latency

## 🔧 Model Management

### Azure ML Integration
Models are synchronized with Azure ML Model Registry:

```python
from azureml.core import Workspace, Model

# Register model in Azure ML
ws = Workspace.from_config()
model = Model.register(
    workspace=ws,
    model_path="models/custom_segmentation/production/best_model.pth",
    model_name="prodsegmenter-v1",
    description="Production grocery shelf segmentation model"
)
```

### Version Control
- Models are versioned using semantic versioning (v1.0.0)
- Major version: Architecture changes
- Minor version: Training data updates  
- Patch version: Hyperparameter tuning

### Storage Management
- Local storage for active development
- Azure Blob Storage for archive and backup
- Azure ML Model Registry for production models

## 🔄 Model Lifecycle

### Development Flow
```
1. Train model → training/train.py
2. Evaluate → training/evaluate.py  
3. Save checkpoint → models/custom_segmentation/{arch}/
4. Test inference → notebooks/model_testing.ipynb
5. Register → Azure ML Model Registry
6. Deploy → deployment/
```

### Production Deployment
```
1. Model validation and testing
2. Performance benchmarking  
3. A/B testing against current production model
4. Gradual rollout with monitoring
5. Full deployment if metrics improve
```

## 📦 Model Optimization

### Optimization Techniques
- **ONNX Export**: Cross-platform deployment
- **TensorRT**: NVIDIA GPU acceleration
- **Quantization**: INT8 for faster inference
- **Pruning**: Reduce model size and latency

### Export Examples
```python
# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "models/production/model.onnx",
    dynamic_axes={'input': {0: 'batch_size'}}
)

# TensorRT optimization
import tensorrt as trt
from torch2trt import torch2trt

model_trt = torch2trt(model, [dummy_input])
torch.jit.save(model_trt, "models/production/model_trt.pth")
```

## 🛡️ Model Security

### Access Control
- Models stored in secure Azure Storage
- Access via managed identities
- Encryption at rest and in transit

### Model Validation
- Checksum verification for model integrity
- Performance regression testing
- Security scanning for malicious code

## 📚 References

- [PyTorch Model Saving](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Azure ML Model Management](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Optimization](https://developer.nvidia.com/tensorrt)

Refer to training and deployment documentation for detailed model usage instructions. 