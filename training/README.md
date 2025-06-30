# Training Directory

This directory contains the core training pipeline for custom segmentation models optimized for real-time grocery shelf segmentation.

## 📁 Directory Structure

```
training/
├── train.py           # Main training script
├── evaluate.py        # Model evaluation and metrics
├── dataset.py         # Dataset classes and data loading
├── models/            # Model architecture definitions
│   ├── __init__.py
│   ├── deeplabv3plus.py
│   ├── unet.py
│   └── yolov8_seg.py
├── utils/             # Training utilities
│   ├── __init__.py
│   ├── losses.py      # Loss functions
│   ├── metrics.py     # Evaluation metrics
│   ├── transforms.py  # Data augmentation
│   └── scheduler.py   # Learning rate scheduling
└── configs/           # Training configurations
    ├── deeplabv3plus.yaml
    ├── unet.yaml
    └── yolov8_seg.yaml
```

## 🎯 Training Pipeline Overview

### Core Scripts

#### `train.py` - Main Training Script
**Purpose**: Orchestrates the complete training process
- Loads data and model configurations
- Initializes model, optimizer, and scheduler
- Executes training loop with validation
- Saves checkpoints and logs metrics
- Integrates with Azure ML for experiment tracking

**Usage**:
```bash
# Train with default config
python train.py --config ../prodsegmenter_config.yaml

# Train specific architecture
python train.py --arch deeplabv3plus --backbone resnet50

# Resume from checkpoint  
python train.py --resume models/checkpoints/epoch_25.pth

# Distributed training
python -m torch.distributed.launch --nproc_per_node=4 train.py --distributed
```

#### `evaluate.py` - Model Evaluation
**Purpose**: Comprehensive model evaluation and benchmarking
- Computes segmentation metrics (mIoU, Dice, Pixel Accuracy)
- Generates confusion matrices and class-wise performance
- Measures inference speed and memory usage
- Creates visualizations and reports

**Usage**:
```bash
# Evaluate trained model
python evaluate.py --model_path models/best_model.pth --data_path ../data/processed

# Benchmark inference speed
python evaluate.py --benchmark --model_path models/best_model.pth

# Generate evaluation report
python evaluate.py --report --output_dir results/
```

#### `dataset.py` - Data Loading
**Purpose**: Dataset classes and data pipeline management
- `GroceryShelfDataset`: Main dataset class for segmentation
- `VideoFrameDataset`: Sequential frame loading for videos
- Data augmentation and preprocessing pipelines
- Multi-threaded data loading with caching

## 🏗️ Model Architectures

### DeepLabV3+ (`models/deeplabv3plus.py`)
**Best for**: High-accuracy segmentation with detailed boundaries
- Atrous Spatial Pyramid Pooling (ASPP)
- Encoder-decoder with skip connections
- Multiple backbone options (ResNet, MobileNet)
- Output stride configuration for speed/accuracy trade-off

**Configuration**:
```yaml
model:
  architecture: "deeplabv3plus"
  backbone: "resnet50"
  num_classes: 10
  output_stride: 16
  aspp_dilate: [12, 24, 36]
  pretrained: true
```

### UNet (`models/unet.py`)
**Best for**: Precise segmentation with efficient training
- Symmetric encoder-decoder architecture
- Skip connections for detail preservation
- Segmentation Models library integration
- Configurable encoder depths and decoder channels

**Configuration**:
```yaml
model:
  architecture: "unet"
  encoder_name: "resnet34"
  encoder_weights: "imagenet"
  decoder_channels: [256, 128, 64, 32, 16]
  num_classes: 10
```

### YOLOv8 Segmentation (`models/yolov8_seg.py`)
**Best for**: Real-time inference (30-60 FPS)
- Instance segmentation with detection
- Optimized for speed and efficiency
- Multiple model sizes (n, s, m, l, x)
- Built-in NMS and confidence filtering

**Configuration**:
```yaml
model:
  architecture: "yolov8_seg"
  model_size: "n"  # n, s, m, l, x
  num_classes: 9   # Excluding background
  confidence_threshold: 0.25
  iou_threshold: 0.45
```

## 🔧 Training Configuration

### Hyperparameters
Key training parameters defined in `prodsegmenter_config.yaml`:

```yaml
training:
  hyperparameters:
    batch_size: 16          # Adjust based on GPU memory
    learning_rate: 0.001    # Initial learning rate
    num_epochs: 100         # Maximum training epochs
    weight_decay: 0.0001    # L2 regularization
    
  optimizer:
    type: "adamw"           # adam, adamw, sgd
    betas: [0.9, 0.999]     # Adam beta parameters
    
  scheduler:
    type: "cosine"          # cosine, step, exponential
    warmup_epochs: 5        # Learning rate warmup
    min_lr: 1e-6           # Minimum learning rate
```

### Data Augmentation
Robust augmentation pipeline for better generalization:

```yaml
training:
  augmentation:
    horizontal_flip: 0.5    # Random horizontal flip
    vertical_flip: 0.1      # Random vertical flip  
    rotation: 15            # Random rotation (degrees)
    brightness: 0.2         # Brightness adjustment
    contrast: 0.2           # Contrast adjustment
    saturation: 0.2         # Saturation adjustment
    hue: 0.1               # Hue adjustment
```

## 📊 Loss Functions and Metrics

### Loss Functions (`utils/losses.py`)
Multiple loss options for different segmentation scenarios:

- **Focal Loss**: Handles class imbalance effectively
- **Cross Entropy**: Standard classification loss
- **Dice Loss**: Overlap-based loss for segmentation
- **Compound Loss**: Combination of multiple losses

### Metrics (`utils/metrics.py`)
Comprehensive evaluation metrics:

- **mIoU (mean Intersection over Union)**: Primary metric
- **Dice Coefficient**: Overlap similarity
- **Pixel Accuracy**: Overall classification accuracy
- **Class-wise IoU**: Per-category performance
- **Inference Speed**: FPS and latency measurements

## 🚀 Azure ML Integration

### Experiment Tracking
Training automatically logs to Azure ML:

```python
from azureml.core import Run

# Get Azure ML run context
run = Run.get_context()

# Log metrics
run.log("train_loss", train_loss)
run.log("val_iou", val_iou)
run.log("learning_rate", current_lr)

# Log images
run.log_image("sample_predictions", prediction_plot)
```

### Compute Cluster Usage
Training leverages Azure ML compute clusters:

```bash
# Submit training job to Azure ML
az ml job create --file training_job.yml --workspace-name ml-prodsegmenter
```

### Model Registration
Best models automatically registered:

```python
# Register best model
model = run.register_model(
    model_name="prodsegmenter",
    model_path="outputs/best_model.pth",
    model_framework="PyTorch",
    description="Grocery shelf segmentation model"
)
```

## 🧪 Development Workflow

### Local Training
```bash
# Setup environment
conda activate prodsegmenter
cd training/

# Quick validation run (1 epoch)
python train.py --config ../prodsegmenter_config.yaml --epochs 1

# Full training run
python train.py --config ../prodsegmenter_config.yaml
```

### Azure ML Training
```bash
# Submit to Azure ML compute cluster
python azure_ml_train.py --config ../prodsegmenter_config.yaml --compute gpu-cluster
```

### Hyperparameter Tuning
```bash
# Grid search with multiple configurations
python hyperparameter_search.py --config_dir configs/ --num_trials 20
```

## 📈 Performance Optimization

### Speed Optimization
- **Mixed Precision Training**: Faster training with AMP
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Data Loading**: Multi-worker data loading and caching
- **Distributed Training**: Multi-GPU scaling

### Memory Optimization
- **Gradient Checkpointing**: Trade computation for memory
- **Model Sharding**: Split large models across GPUs
- **Dynamic Batching**: Adjust batch size based on image size

## 🔍 Monitoring and Debugging

### Training Monitoring
```python
# Watch training progress
tensorboard --logdir=logs/

# Azure ML Studio monitoring
# Navigate to Azure ML workspace → Experiments → View run details
```

### Common Issues and Solutions

**CUDA Out of Memory**:
```bash
# Reduce batch size
python train.py --batch_size 8

# Enable gradient accumulation
python train.py --accumulate_grad_batches 2
```

**Slow Convergence**:
```bash
# Adjust learning rate
python train.py --learning_rate 0.01

# Use different optimizer
python train.py --optimizer sgd
```

**Poor Validation Performance**:
```bash
# Increase data augmentation
python train.py --augmentation_strength 0.8

# Add regularization
python train.py --weight_decay 0.001
```

## 📋 Quality Assurance

### Pre-training Checklist
- [ ] Data preprocessing completed
- [ ] Train/validation splits created
- [ ] Configuration file validated
- [ ] GPU availability confirmed
- [ ] Azure ML workspace connected

### Post-training Validation
- [ ] Model checkpoints saved
- [ ] Metrics logged and visualized
- [ ] Inference speed benchmarked
- [ ] Model registered in Azure ML
- [ ] Evaluation report generated

## 🔗 Integration Points

### With Data Pipeline
- Loads processed data from `data/processed/`
- Uses manifests created by preprocessing notebooks
- Applies augmentations defined in config

### With Deployment
- Saves models in deployment-ready format
- Exports ONNX/TensorRT optimized versions
- Validates real-time performance requirements

### With Testing
- Integration with `tests/test_train.py`
- Automated smoke tests for training pipeline
- Performance regression testing

Run `python train.py --help` for complete usage instructions and options. 