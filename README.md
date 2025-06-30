# ProdSegmenter

**Real-time semantic segmentation of products on grocery store shelves using Azure-native cloud infrastructure.**

[![Build Status](https://github.com/your-org/prodsegmenter/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/your-org/prodsegmenter/actions)
[![Azure ML](https://img.shields.io/badge/Azure-ML%20Ready-blue)](https://azure.microsoft.com/en-us/services/machine-learning/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Overview

ProdSegmenter is a full-stack computer vision system that performs **real-time semantic segmentation** of products on grocery store shelves from video input (30–60 FPS). The system identifies product regions, overlays segmented output on video streams, and provides a web-based interface for visualization and interaction.

### Key Features

- **🚀 Real-time Processing**: 30-60 FPS video segmentation
- **☁️ Azure-Native**: Fully hosted on Microsoft Azure cloud platform  
- **📈 Scalable Architecture**: Container-based deployment with auto-scaling
- **🎨 Web Interface**: Interactive Streamlit frontend for video upload and visualization
- **🔄 MLOps Pipeline**: Automated model training, evaluation, and deployment
- **🏗️ Production Ready**: Comprehensive testing, monitoring, and CI/CD

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        A[👤 User] --> B[🎥 Video Upload/Stream]
        B --> C[📱 Streamlit Frontend]
    end
    
    subgraph "Azure Cloud Infrastructure"
        C --> D[⚡ Azure Function API]
        D --> E[🧠 ML Model]
        E --> F[📊 Segmentation Results]
        F --> C
        
        G[💾 Azure Blob Storage] --> E
        H[🏋️ Azure ML] --> G
        I[📈 Azure Monitor] --> D
        J[🔧 Azure Container Registry] --> D
    end
    
    subgraph "Development Pipeline"
        K[👨‍💻 Developer] --> L[🐙 GitHub]
        L --> M[🔄 GitHub Actions]
        M --> N[🧪 Testing]
        N --> O[🏗️ Build Images]
        O --> J
        M --> P[📚 Model Training]
        P --> H
    end
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style G fill:#fce4ec
    style H fill:#e0f2f1
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit Frontend
    participant F as Azure Function
    participant M as ML Model
    participant B as Blob Storage
    participant Mon as Azure Monitor
    
    U->>S: Upload Video (30-60 FPS)
    S->>F: POST /api/segment_video
    F->>B: Load Model Weights
    B-->>F: Model Artifacts
    F->>M: Initialize Model
    
    loop For Each Frame
        S->>F: POST /api/segment_frame
        F->>M: Process Frame
        M-->>F: Segmentation Mask
        F->>Mon: Log Metrics (Latency, FPS)
        F-->>S: Return Segmented Frame
        S->>U: Display Overlay
    end
    
    S->>U: Download Processed Video
```

---

## ⚙️ Technology Stack

### Infrastructure & Compute
| Component | Technology | Purpose | SLA/Performance |
|-----------|------------|---------|-----------------|
| **Training** | Azure Machine Learning | Model training & experiments | 99.9% uptime |
| **Inference** | Azure Functions | Real-time API endpoints | <33ms latency |
| **Storage** | Azure Blob Storage | Data & model artifacts | 99.999% durability |
| **Frontend** | Azure App Service | Streamlit web interface | Auto-scaling |
| **Registry** | Azure Container Registry | Docker image management | Global replication |
| **Monitoring** | Azure Application Insights | Performance & error tracking | Real-time alerts |

### Machine Learning Stack
| Component | Technology | Use Case | Performance Target |
|-----------|------------|----------|-------------------|
| **Bootstrapping** | Segment Anything Model (SAM) | Initial mask generation | High quality masks |
| **Architecture** | DeepLabV3+/UNet/YOLOv8-Seg | Real-time segmentation | 30-60 FPS |
| **Framework** | PyTorch 2.0+ | Model development | GPU acceleration |
| **Optimization** | ONNX/TensorRT | Inference acceleration | <17ms per frame |
| **Tracking** | MLflow + Azure ML | Experiment management | Version control |

### Development & Deployment
| Tool | Purpose | Integration |
|------|---------|-------------|
| **GitHub Actions** | CI/CD pipeline | Automated testing & deployment |
| **Conda** | Environment management | Reproducible dependencies |
| **Pytest** | Testing framework | 80%+ code coverage |
| **Black/Flake8** | Code quality | Automated formatting |
| **Docker** | Containerization | Multi-stage builds |

---

## 📊 Performance Requirements

### Real-time Processing Targets

| Model Architecture | Input Size | Target FPS | Latency (ms) | mIoU | GPU Memory |
|-------------------|------------|------------|---------------|------|------------|
| **YOLOv8n-seg** | 640×640 | 60+ | <17 | 0.75+ | 2GB |
| **YOLOv8s-seg** | 640×640 | 45+ | <22 | 0.78+ | 3GB |
| **UNet-ResNet34** | 512×512 | 35+ | <28 | 0.80+ | 4GB |
| **DeepLabV3+-ResNet50** | 512×512 | 30+ | <33 | 0.82+ | 6GB |

### System Scalability

```mermaid
graph LR
    subgraph "Load Balancing"
        A[👥 100+ Users] --> B[⚖️ Load Balancer]
    end
    
    subgraph "Auto-scaling"
        B --> C[🔧 Function App 1]
        B --> D[🔧 Function App 2]
        B --> E[🔧 Function App N]
    end
    
    subgraph "Performance Monitoring"
        C --> F[📊 App Insights]
        D --> F
        E --> F
        F --> G[🚨 Alerts]
    end
    
    style A fill:#e3f2fd
    style B fill:#f1f8e9
    style F fill:#fff3e0
```

---

## 📁 Project Structure

```
prodsegmenter/
├── 📄 README.md                    # Project overview and setup
├── 📄 environment.yml              # Conda environment
├── 📄 prodsegmenter_config.yaml   # Project configuration
├── 📄 .gitignore                   # Git ignore rules
│
├── 📂 data/                        # Data storage and management
│   ├── raw/                        # Raw video files
│   ├── processed/                  # Processed frames and masks
│   └── README.md                   # Data documentation
│
├── 📂 notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_download_dataset.ipynb   # Data acquisition
│   ├── 02_sam_bootstrap.ipynb      # SAM mask generation
│   ├── 03_preprocessing.ipynb      # Data preprocessing
│   └── README.md                   # Notebook documentation
│
├── 📂 models/                      # Model storage
│   ├── sam/                        # SAM model weights
│   ├── custom_segmentation/        # Custom trained models
│   └── README.md                   # Model documentation
│
├── 📂 training/                    # Training pipeline
│   ├── train.py                    # Model training script
│   ├── evaluate.py                 # Model evaluation
│   ├── dataset.py                  # Dataset classes
│   └── README.md                   # Training documentation
│
├── 📂 deployment/                  # Deployment assets
│   ├── azure_function/             # Azure Function API
│   ├── streamlit_frontend/         # Web interface
│   └── README.md                   # Deployment documentation
│
├── 📂 tests/                       # Test suite
│   ├── test_train.py               # Training tests
│   ├── test_inference.py           # Inference tests
│   └── README.md                   # Testing documentation
│
└── 📂 .github/                     # CI/CD workflows
    └── workflows/
        └── ci.yml                  # GitHub Actions CI
```

---

## 🚀 Quick Start Guide

### Prerequisites

- **Azure Subscription** with appropriate permissions
- **Git** installed locally
- **Conda/Miniconda** installed
- **Azure CLI** installed and configured

### 1. 🔧 Azure Environment Setup

```bash
# Login to Azure
az login

# Set your subscription
az account set --subscription "your-subscription-id"

# Create resource group
az group create --name rg-prodsegmenter --location eastus

# Create Azure ML workspace
az ml workspace create --name ml-prodsegmenter --resource-group rg-prodsegmenter
```

### 2. 📥 Project Setup

```bash
# Clone the repository
git clone https://github.com/your-org/prodsegmenter.git
cd prodsegmenter

# Create and activate conda environment
conda env create -f environment.yml
conda activate prodsegmenter

# Initialize git (if not cloned)
git init
git add .
git commit -m "Initial project setup"
```

### 3. ⚙️ Configuration

```bash
# Copy configuration template
cp prodsegmenter_config.yaml.template prodsegmenter_config.yaml

# Edit configuration with your Azure details
nano prodsegmenter_config.yaml
```

Set the following environment variables:
```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="rg-prodsegmenter"
export AZURE_ML_WORKSPACE="ml-prodsegmenter"
export AZURE_STORAGE_ACCOUNT="stprodsegmenter"
```

### 4. 🧪 Verification

```bash
# Test environment setup
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test Azure connectivity
az ml workspace show --name ml-prodsegmenter --resource-group rg-prodsegmenter
```

---

## 📈 Development Workflow

### Phase-by-Phase Implementation

```mermaid
gantt
    title ProdSegmenter Development Timeline
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Project Setup           :done, setup, 2024-01-01, 2024-01-03
    Data Acquisition        :active, data, 2024-01-04, 2024-01-06
    SAM Bootstrapping       :sam, 2024-01-07, 2024-01-09
    
    section Phase 2: Model Development
    Data Preprocessing      :prep, 2024-01-10, 2024-01-12
    Model Training          :train, 2024-01-13, 2024-01-17
    Model Evaluation        :eval, 2024-01-18, 2024-01-20
    
    section Phase 3: Deployment
    API Development         :api, 2024-01-21, 2024-01-24
    Frontend Development    :frontend, 2024-01-25, 2024-01-27
    Testing & CI/CD         :testing, 2024-01-28, 2024-01-30
    
    section Phase 4: Production
    Monitoring Setup        :monitor, 2024-01-31, 2024-02-02
    Performance Optimization :optimize, 2024-02-03, 2024-02-05
    Documentation           :docs, 2024-02-06, 2024-02-07
```

### 🔄 MLOps Pipeline

```mermaid
flowchart LR
    subgraph "Data Pipeline"
        A[📹 Raw Videos] --> B[🔄 Preprocessing]
        B --> C[🏷️ SAM Labeling]
        C --> D[📊 Training Data]
    end
    
    subgraph "Model Pipeline"
        D --> E[🏋️ Training]
        E --> F[📏 Evaluation]
        F --> G{✅ Quality Gate}
        G -->|Pass| H[📦 Model Registry]
        G -->|Fail| E
    end
    
    subgraph "Deployment Pipeline"
        H --> I[🚀 Staging Deploy]
        I --> J[🧪 Integration Tests]
        J --> K{✅ Test Gate}
        K -->|Pass| L[🌐 Production Deploy]
        K -->|Fail| I
    end
    
    subgraph "Monitoring"
        L --> M[📊 Performance Metrics]
        M --> N[🚨 Alerts]
        N --> O{📉 Drift Detected?}
        O -->|Yes| E
        O -->|No| M
    end
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style L fill:#fff3e0
    style N fill:#ffebee
```

---

## 🧪 Development Commands

### Data Processing
```bash
# Extract and preprocess data
jupyter lab notebooks/01_download_dataset.ipynb

# Generate SAM masks
jupyter lab notebooks/02_sam_bootstrap.ipynb

# Prepare training data
jupyter lab notebooks/03_preprocessing.ipynb
```

### Model Training
```bash
# Local training (quick test)
cd training/
python train.py --config ../prodsegmenter_config.yaml --epochs 5

# Azure ML training (full)
python train.py --config ../prodsegmenter_config.yaml --azure
```

### Testing
```bash
# Run all tests
pytest tests/ -v --cov=training --cov=deployment

# Run specific test categories
pytest tests/test_train.py -v
pytest tests/test_inference.py -v
```

### Deployment
```bash
# Local development
cd deployment/streamlit_frontend/
streamlit run app.py

# Azure deployment
cd deployment/azure_function/
func azure functionapp publish func-prodsegmenter-inference
```

---

## 📊 Monitoring & Analytics

### Key Metrics Dashboard

| Metric Category | Key Indicators | Target Values | Alerting |
|----------------|----------------|---------------|----------|
| **Performance** | Response Time, FPS, Throughput | <33ms, >30 FPS, >100 req/s | >50ms response |
| **Accuracy** | mIoU, Pixel Accuracy, F1 Score | >0.80, >0.90, >0.85 | <0.75 mIoU |
| **Reliability** | Uptime, Error Rate, Success Rate | >99.9%, <1%, >99% | >5% error rate |
| **Resource** | CPU, Memory, GPU Utilization | <80%, <85%, <90% | >95% utilization |

### Architecture Decision Records (ADRs)

```mermaid
graph TD
    A[🤔 Architecture Decision] --> B{🎯 Real-time Requirement?}
    B -->|Yes| C[⚡ Azure Functions]
    B -->|No| D[🏗️ Azure Container Apps]
    
    C --> E{📊 Model Size?}
    E -->|Small| F[🚀 YOLOv8n-seg]
    E -->|Large| G[🧠 DeepLabV3+]
    
    D --> H{💰 Cost Priority?}
    H -->|High| I[📦 Consumption Plan]
    H -->|Low| J[🏃 Premium Plan]
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#f3e5f5
```

---

## 🤝 Contributing

### Development Process

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-awesome-improvement`
3. **Make** changes and add tests
4. **Run** the test suite: `pytest tests/ --cov=80`
5. **Format** code: `black . && flake8 .`
6. **Submit** a pull request with detailed description

### Code Quality Standards

- **Test Coverage**: Minimum 80% overall, 95% for core modules
- **Documentation**: All public functions must have docstrings
- **Performance**: Real-time functions must meet latency requirements
- **Security**: All inputs must be validated and sanitized

---

## 📚 Additional Resources

### Documentation Links
- [📖 Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [🔧 Azure Functions Guide](https://docs.microsoft.com/en-us/azure/azure-functions/)
- [🎨 Streamlit Documentation](https://docs.streamlit.io/)
- [🤖 SAM Model Documentation](https://github.com/facebookresearch/segment-anything)

### Learning Resources
- [🎓 Computer Vision Course](https://cs231n.stanford.edu/)
- [☁️ Azure ML Learning Path](https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-machine-learning-service/)
- [🐍 PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Contact

- **📧 Email**: support@prodsegmenter.com
- **💬 Discord**: [ProdSegmenter Community](https://discord.gg/prodsegmenter)
- **📚 Wiki**: [Documentation](https://github.com/your-org/prodsegmenter/wiki)
- **🐛 Issues**: [GitHub Issues](https://github.com/your-org/prodsegmenter/issues)

---

## 🎯 Current Status

**✅ Prompt 0 & 1 Complete**: Architecture defined, project bootstrapped, ready for data acquisition

**🚀 Next Steps**: 
- Proceed to Prompt 2: Download and organize grocery shelf datasets
- Set up SKU110K dataset processing
- Begin SAM bootstrap pipeline

---

*Built with ❤️ by the ProdSegmenter team using Azure cloud infrastructure* # ProdSegmenter
