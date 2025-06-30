# Notebooks Directory

This directory contains Jupyter notebooks for data exploration, preprocessing, and model development workflows.

## 📓 Notebook Overview

### 01_download_dataset.ipynb
**Purpose**: Data acquisition and initial dataset setup
- Download public grocery shelf datasets
- Set up Azure Blob Storage connections
- Organize raw data directory structure
- Validate data quality and formats

**Outputs**: Raw videos and images in `data/raw/`

### 02_sam_bootstrap.ipynb  
**Purpose**: Generate initial segmentation masks using Segment Anything Model (SAM)
- Load and configure SAM model (ViT-H/L/B variants)
- Process video frames through SAM
- Generate high-quality segmentation masks
- Create training labels for custom model

**Outputs**: Segmentation masks in `data/processed/masks/`

### 03_preprocessing.ipynb
**Purpose**: Data preprocessing and preparation for training
- Extract frames from videos at target FPS
- Resize and normalize images
- Create train/validation/test splits
- Generate data augmentation examples
- Create dataset manifests and metadata

**Outputs**: Processed data ready for training

## 🔄 Workflow Sequence

Run notebooks in this order for optimal results:

```
01_download_dataset.ipynb
        ↓
02_sam_bootstrap.ipynb  
        ↓
03_preprocessing.ipynb
        ↓
Training Pipeline (training/ directory)
```

## 🛠️ Setup Requirements

### Environment Setup
```bash
# Activate the conda environment
conda activate prodsegmenter

# Launch Jupyter Lab
jupyter lab

# Navigate to notebooks/ directory
```

### Azure Configuration
Ensure these environment variables are set:
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="rg-prodsegmenter"
export AZURE_STORAGE_ACCOUNT="stprodsegmenter"
```

### GPU Requirements
- **SAM Bootstrap**: Requires GPU with 8GB+ VRAM
- **Preprocessing**: Can run on CPU or GPU
- **Data Download**: CPU-only operations

## 📊 Expected Outputs

### Data Metrics
After running all notebooks, you should have:
- 1,000-10,000 video frames
- High-quality SAM masks for each frame
- Organized train/val/test splits (70/20/10)
- Data quality reports and statistics

### File Sizes
- Raw videos: 1-10GB total
- Processed frames: 2-20GB total  
- SAM masks: 500MB-5GB total
- Notebooks outputs: 100-500MB

## 🧪 Development Tips

### Running Individual Cells
- Use `Shift+Enter` to run cells sequentially
- Monitor GPU memory usage in SAM notebook
- Check output directory sizes regularly

### Debugging Common Issues
- **CUDA Out of Memory**: Reduce batch size in SAM processing
- **Slow Processing**: Use smaller SAM model variant (ViT-B)
- **File Not Found**: Check data paths in config file

### Performance Optimization
- Process videos in batches for memory efficiency
- Use multiprocessing for frame extraction
- Cache SAM model weights locally

## 📋 Quality Checks

### Data Validation
Each notebook includes validation steps:
- File format verification
- Image quality checks  
- Mask quality assessment
- Data distribution analysis

### Output Verification
```python
# Check notebook outputs
from pathlib import Path

# Verify frame extraction
frames = list(Path('data/processed/frames').glob('*.jpg'))
print(f"Extracted frames: {len(frames)}")

# Verify mask generation  
masks = list(Path('data/processed/masks').glob('*.png'))
print(f"Generated masks: {len(masks)}")

# Check data splits
import json
with open('data/processed/train_split.json') as f:
    train_data = json.load(f)
print(f"Training samples: {len(train_data)}")
```

## 🔗 Integration Points

### With Training Pipeline
- Notebooks prepare data for `training/dataset.py`
- Generated manifests used by data loaders
- Preprocessing parameters saved to config

### With Azure ML
- Datasets uploaded to Blob Storage
- Experiment tracking via MLflow
- Compute cluster utilization for large datasets

## 📚 Additional Resources

- [SAM Documentation](https://github.com/facebookresearch/segment-anything)
- [Azure ML Notebooks Guide](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Jupyter Best Practices](https://jupyter.readthedocs.io/en/latest/content-quickstart.html)

Run the notebooks sequentially and refer to this README for troubleshooting and optimization tips. 