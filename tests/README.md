# Tests Directory

Comprehensive test suite for the ProdSegmenter project covering training, inference, and deployment components.

## 📁 Test Organization

```
tests/
├── test_train.py           # Training pipeline tests
├── test_inference.py       # Model inference tests
├── test_dataset.py         # Data loading and processing tests
├── test_models.py          # Model architecture tests
├── test_deployment.py      # Deployment and API tests
├── conftest.py            # Pytest configuration and fixtures
├── fixtures/              # Test data and fixtures
│   ├── sample_images/     # Sample test images
│   ├── sample_videos/     # Sample test videos
│   └── mock_models/       # Mock model weights
└── integration/           # Integration tests
    ├── test_e2e_pipeline.py
    └── test_azure_integration.py
```

## 🎯 Testing Strategy

### Unit Tests
- **Model Architecture**: Verify model construction and forward pass
- **Data Processing**: Test dataset loading, transforms, augmentation
- **Training Components**: Validate loss functions, metrics, optimizers
- **Utility Functions**: Test preprocessing, postprocessing utilities

### Integration Tests  
- **End-to-End Pipeline**: Complete workflow from data to deployment
- **Azure Integration**: Test Azure ML, Storage, Functions integration
- **API Testing**: Validate REST endpoints and responses
- **Performance Testing**: Benchmark inference speed and accuracy

### Test Coverage Requirements
- **Minimum Coverage**: 80% overall code coverage
- **Critical Components**: 95% coverage for core training/inference
- **Documentation**: All public functions have docstring tests

## 🧪 Test Execution

### Local Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_train.py -v

# Run with coverage
pytest tests/ --cov=training --cov=deployment --cov-report=html

# Run fast tests only
pytest tests/ -m "not slow"

# Run integration tests
pytest tests/integration/ -v
```

### Azure DevOps Integration
```yaml
# test-pipeline.yml
trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.10'

- script: |
    pip install -r requirements.txt
    pip install pytest pytest-cov pytest-mock
  displayName: 'Install dependencies'

- script: |
    pytest tests/ --junitxml=test-results.xml --cov-report=xml
  displayName: 'Run tests'

- task: PublishTestResults@2
  inputs:
    testResultsFiles: 'test-results.xml'
    testRunTitle: 'Python Tests'

- task: PublishCodeCoverageResults@1
  inputs:
    codeCoverageTool: 'Cobertura'
    summaryFileLocation: 'coverage.xml'
```

## 🔧 Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    gpu: marks tests that require GPU
    azure: marks tests that require Azure services
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### Test Fixtures (conftest.py)
```python
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_image():
    """Sample RGB image for testing"""
    return torch.randn(3, 512, 512)

@pytest.fixture
def sample_mask():
    """Sample segmentation mask"""
    return torch.randint(0, 10, (512, 512))

@pytest.fixture
def mock_model():
    """Mock segmentation model"""
    class MockModel(torch.nn.Module):
        def forward(self, x):
            return torch.randn(x.shape[0], 10, x.shape[2], x.shape[3])
    return MockModel()

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'batch_size': 2,
        'num_classes': 10,
        'learning_rate': 0.001
    }
```

## 📊 Test Categories

### Training Tests (test_train.py)
- Training loop execution
- Loss computation and backpropagation
- Optimizer and scheduler functionality
- Checkpoint saving and loading
- Multi-GPU training support

### Inference Tests (test_inference.py)  
- Model loading and inference
- Batch processing capabilities
- Real-time performance benchmarks
- ONNX/TensorRT optimization
- Memory usage validation

### Dataset Tests (test_dataset.py)
- Data loading and iteration
- Augmentation pipeline
- Train/validation splits
- Data format validation
- Batch collation

### Model Tests (test_models.py)
- Architecture instantiation
- Forward pass computation
- Parameter counting
- Model serialization
- Multiple backbone support

### Deployment Tests (test_deployment.py)
- Azure Function endpoints
- Streamlit app functionality
- API request/response validation
- Container deployment
- Health check endpoints

## 🚀 Continuous Integration

### GitHub Actions Workflow
```yaml
name: CI/CD Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=./ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 🔍 Performance Testing

### Benchmark Tests
```python
# test_performance.py
def test_inference_speed(model, sample_image):
    """Test real-time inference requirements"""
    import time
    
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(sample_image.unsqueeze(0))
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            _ = model(sample_image.unsqueeze(0))
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        fps = 1.0 / avg_time
        
        assert fps >= 30, f"Model too slow: {fps:.1f} FPS < 30 FPS"
```

### Memory Testing
```python
def test_memory_usage(model, sample_batch):
    """Test GPU memory usage within limits"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model.cuda()
        
        output = model(sample_batch.cuda())
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
        assert memory_mb < 8000, f"Memory usage too high: {memory_mb:.1f} MB"
```

## 🔧 Mock Services

### Azure Service Mocks
```python
# conftest.py
@pytest.fixture
def mock_azure_storage():
    """Mock Azure Blob Storage"""
    with patch('azure.storage.blob.BlobServiceClient') as mock:
        yield mock

@pytest.fixture  
def mock_azure_ml():
    """Mock Azure ML workspace"""
    with patch('azureml.core.Workspace') as mock:
        yield mock
```

## 📋 Test Maintenance

### Regular Test Updates
- Update tests when adding new features
- Maintain test data currency
- Review and update performance benchmarks
- Ensure Azure service mocks stay current

### Test Data Management
- Keep test datasets small but representative
- Version control test fixtures
- Automated test data generation where possible
- Regular cleanup of temporary test files

## 🛡️ Security Testing

### Input Validation Tests
```python
def test_malicious_input_handling():
    """Test handling of malicious inputs"""
    # Test oversized inputs
    # Test malformed data
    # Test injection attempts
    pass

def test_authentication():
    """Test API authentication"""
    # Test valid/invalid tokens
    # Test authorization levels
    pass
```

Run `pytest --help` for complete testing options and configuration. 