# Deployment Directory

This directory contains all deployment assets for the ProdSegmenter real-time inference system, including Azure Functions API and Streamlit frontend.

## 📁 Directory Structure

```
deployment/
├── azure_function/           # Azure Functions inference API
│   ├── function_app.py      # Main Azure Function app
│   ├── model_handler.py     # Model loading and inference
│   ├── requirements.txt     # Python dependencies
│   ├── host.json           # Function runtime configuration
│   ├── local.settings.json # Local development settings
│   └── Dockerfile          # Container configuration
├── streamlit_frontend/      # Web interface
│   ├── app.py              # Main Streamlit application
│   ├── components/         # UI components
│   │   ├── video_uploader.py
│   │   ├── model_selector.py
│   │   └── results_viewer.py
│   ├── utils/              # Frontend utilities
│   │   ├── api_client.py   # API communication
│   │   ├── video_processor.py
│   │   └── visualization.py
│   ├── requirements.txt    # Streamlit dependencies
│   └── Dockerfile          # Container configuration
├── scoring/                # Model scoring scripts
│   ├── score.py           # Azure ML scoring script
│   ├── inference_config.json
│   └── environment.yml
└── infrastructure/         # Infrastructure as Code
    ├── main.bicep         # Azure Bicep templates
    ├── parameters.json    # Deployment parameters
    └── deploy.sh          # Deployment script
```

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │ -> │  Azure Function │ -> │  Azure Blob     │
│   Frontend      │    │  Inference API  │    │  Storage        │
│  (Port 8501)    │    │  (Port 80/443)  │    │  (Models)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  App Service    │    │ Container Apps  │    │   Azure ML      │
│   (Hosting)     │    │   (Scaling)     │    │  (Registry)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **User Upload**: Video uploaded via Streamlit frontend
2. **API Request**: Frontend sends video to Azure Function
3. **Model Inference**: Function processes frames with trained model
4. **Results Return**: Segmented video returned to frontend
5. **Visualization**: Results displayed with overlay annotations

## 🚀 Azure Functions API (`azure_function/`)

### Core Components

#### `function_app.py` - Main Function App
**Purpose**: HTTP-triggered Azure Function for model inference
- Receives video upload requests
- Handles authentication and validation
- Orchestrates inference pipeline
- Returns segmented results

**Endpoints**:
```
POST /api/segment_video    # Process uploaded video
GET  /api/health          # Health check endpoint
GET  /api/models          # Available model information
POST /api/segment_frame   # Process single frame
```

#### `model_handler.py` - Model Management
**Purpose**: Handles model loading, caching, and inference
- Lazy model loading with caching
- Multi-model support (DeepLabV3+, UNet, YOLO)
- ONNX/TensorRT optimization
- Memory management for concurrent requests

### Performance Specifications
- **Target Latency**: <33ms per frame (30 FPS)
- **Throughput**: 100+ concurrent requests
- **Model Size**: <500MB for fast loading
- **Memory Usage**: <2GB per instance

### Local Development
```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Navigate to function directory
cd deployment/azure_function/

# Install dependencies
pip install -r requirements.txt

# Start local development server
func start --python
```

### Deployment
```bash
# Deploy to Azure Functions
az functionapp create --resource-group rg-prodsegmenter \
                     --name func-prodsegmenter-inference \
                     --storage-account stprodsegmenter \
                     --functions-version 4 \
                     --runtime python \
                     --runtime-version 3.10

# Deploy function code
func azure functionapp publish func-prodsegmenter-inference
```

## 🎨 Streamlit Frontend (`streamlit_frontend/`)

### Application Structure

#### `app.py` - Main Application
**Purpose**: Central Streamlit application with navigation and layout
- Multi-page navigation
- Session state management
- Global error handling
- Azure authentication integration

#### UI Components (`components/`)

**`video_uploader.py`**:
- Drag-and-drop video upload
- Format validation and preview
- Progress tracking
- File size optimization

**`model_selector.py`**:
- Model architecture selection
- Configuration parameter tuning
- Performance vs accuracy trade-offs
- Real-time parameter updates

**`results_viewer.py`**:
- Side-by-side video comparison
- Segmentation overlay controls
- Download processed videos
- Performance metrics display

### Frontend Features
- **Real-time Processing**: Live video segmentation preview
- **Interactive Controls**: Adjust confidence thresholds, colors
- **Batch Processing**: Upload multiple videos simultaneously
- **Export Options**: Download results in various formats
- **Analytics Dashboard**: Processing history and statistics

### Local Development
```bash
# Navigate to frontend directory
cd deployment/streamlit_frontend/

# Install dependencies
pip install -r requirements.txt

# Start Streamlit app
streamlit run app.py

# Access at http://localhost:8501
```

### Deployment to Azure App Service
```bash
# Create App Service plan
az appservice plan create --name asp-prodsegmenter \
                         --resource-group rg-prodsegmenter \
                         --sku B1 \
                         --is-linux

# Create web app
az webapp create --resource-group rg-prodsegmenter \
                --plan asp-prodsegmenter \
                --name app-prodsegmenter-frontend \
                --deployment-container-image-name prodsegmenter-frontend:latest
```

## 📊 Scoring Scripts (`scoring/`)

### Azure ML Endpoint Deployment

#### `score.py` - Scoring Script
**Purpose**: Azure ML managed endpoint for batch processing
- Batch inference for large datasets
- Managed scaling and monitoring
- Integration with Azure ML pipeline
- Cost-optimized for non-real-time processing

```python
def init():
    """Initialize model and dependencies"""
    global model
    model_path = Model.get_model_path("prodsegmenter")
    model = load_model(model_path)

def run(data):
    """Process inference request"""
    frames = preprocess_input(data)
    predictions = model(frames)
    return postprocess_output(predictions)
```

### Deployment Configuration
```yaml
# inference_config.json
{
  "entryScript": "score.py",
  "runtime": "python",
  "condaFile": "environment.yml",
  "extraDockerFileSteps": "RUN apt-get update && apt-get install -y libgl1-mesa-glx"
}
```

## 🏢 Infrastructure as Code (`infrastructure/`)

### Azure Bicep Templates (`main.bicep`)
**Purpose**: Automated Azure resource provisioning
- Resource group and storage accounts
- Azure ML workspace and compute
- Function apps and app services
- Networking and security configuration
- Monitoring and logging setup

### Deployment Parameters (`parameters.json`)
```json
{
  "projectName": "prodsegmenter",
  "location": "eastus",
  "environment": "prod",
  "functionAppSku": "EP1",
  "appServiceSku": "B1",
  "storageAccountType": "Standard_LRS"
}
```

### Automated Deployment (`deploy.sh`)
```bash
#!/bin/bash
# Complete infrastructure deployment
az deployment group create \
  --resource-group rg-prodsegmenter \
  --template-file main.bicep \
  --parameters @parameters.json
```

## 🔧 Configuration and Setup

### Environment Variables
Required environment variables for deployment:

```bash
# Azure credentials
AZURE_SUBSCRIPTION_ID="your-subscription-id"
AZURE_TENANT_ID="your-tenant-id"
AZURE_CLIENT_ID="your-client-id"
AZURE_CLIENT_SECRET="your-client-secret"

# Storage configuration  
AZURE_STORAGE_ACCOUNT="stprodsegmenter"
AZURE_STORAGE_KEY="storage-access-key"

# Function app settings
FUNCTION_APP_NAME="func-prodsegmenter-inference"
WEBSITE_RUN_FROM_PACKAGE="1"

# Model configuration
MODEL_NAME="prodsegmenter-v1"
MODEL_VERSION="1"
```

### Security Configuration
- **Authentication**: Azure AD integration
- **HTTPS Only**: Force SSL for all endpoints
- **CORS**: Configure allowed origins
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize all inputs

## 📈 Monitoring and Scaling

### Application Insights Integration
```python
from applicationinsights import TelemetryClient

# Track custom metrics
tc = TelemetryClient(os.environ['APPINSIGHTS_INSTRUMENTATION_KEY'])
tc.track_metric("inference_latency", latency_ms)
tc.track_metric("concurrent_requests", request_count)
```

### Auto-scaling Configuration
```yaml
# Function app scaling
scaleSettings:
  minimumElasticInstanceCount: 1
  maximumElasticInstanceCount: 20
  functionsScaleLimit: 200

# App service scaling  
autoScaleSettings:
  minCapacity: 1
  maxCapacity: 10
  defaultCapacity: 2
  rules:
    - metricName: "CpuPercentage"
      operator: "GreaterThan"
      threshold: 70
      action: "Increase"
```

## 🧪 Testing and Validation

### Load Testing
```bash
# Install testing tools
pip install locust

# Run load tests
locust -f load_tests.py --host=https://func-prodsegmenter-inference.azurewebsites.net

# Monitor during testing
az monitor metrics alert create --name "High Response Time" \
                               --resource-group rg-prodsegmenter \
                               --condition "avg Percentage CPU > 80"
```

### Integration Testing
```python
# Test API endpoints
import requests

# Test video processing
response = requests.post(
    "https://func-prodsegmenter-inference.azurewebsites.net/api/segment_video",
    files={"video": open("test_video.mp4", "rb")}
)
assert response.status_code == 200
assert "segmented_video" in response.json()
```

## 🔄 CI/CD Integration

### Deployment Pipeline
```yaml
# Azure DevOps pipeline
stages:
  - stage: Build
    jobs:
      - job: BuildFunction
        steps:
          - task: UsePythonVersion@0
          - script: pip install -r requirements.txt
          - task: ArchiveFiles@2

  - stage: Deploy
    jobs:
      - deployment: DeployFunction
        environment: production
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureFunctionApp@1
```

### Health Checks
```python
# Automated health monitoring
def health_check():
    """Verify all components are operational"""
    checks = {
        "function_app": test_function_endpoint(),
        "model_loading": test_model_inference(),
        "storage_access": test_blob_storage(),
        "frontend": test_streamlit_app()
    }
    return all(checks.values())
```

## 📚 Additional Resources

- [Azure Functions Python Developer Guide](https://docs.microsoft.com/en-us/azure/azure-functions/functions-reference-python)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Azure App Service Deployment](https://docs.microsoft.com/en-us/azure/app-service/)
- [Azure ML Endpoint Deployment](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-managed-online-endpoints)

For deployment instructions and troubleshooting, refer to the specific component documentation and Azure portal monitoring. 