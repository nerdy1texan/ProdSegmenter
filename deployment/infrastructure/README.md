# Infrastructure as Code (IaC)

This directory contains Infrastructure as Code (IaC) templates and scripts for deploying ProdSegmenter on Microsoft Azure.

## 🏗️ Infrastructure Overview

The ProdSegmenter infrastructure includes:

- **Azure Machine Learning Workspace** - Model training and experiment tracking
- **Azure Storage Account** - Data and model artifact storage
- **Azure Functions** - Real-time inference API endpoints
- **Azure App Service** - Streamlit frontend hosting  
- **Azure Container Registry** - Custom Docker image storage
- **Azure Application Insights** - Monitoring and telemetry
- **Azure Key Vault** - Secrets and configuration management

## 📁 Files Structure

```
deployment/infrastructure/
├── azure-resources.bicep          # Main Bicep IaC template
├── parameters.json                # Parameter file for deployment
├── deploy.sh                     # Deployment automation script
├── README.md                     # This file
└── environments/
    ├── dev.parameters.json       # Development environment parameters
    ├── staging.parameters.json   # Staging environment parameters  
    └── prod.parameters.json      # Production environment parameters
```

## 🚀 Quick Deployment

### Prerequisites

1. **Azure CLI** installed and configured
2. **Azure subscription** with appropriate permissions
3. **Bicep CLI** (automatically installed with Azure CLI 2.20.0+)

### One-Command Deployment

```bash
# Deploy to development environment
./deploy.sh dev

# Deploy to staging environment  
./deploy.sh staging

# Deploy to production environment
./deploy.sh prod
```

### Manual Deployment

If you prefer manual control:

```bash
# 1. Login to Azure
az login

# 2. Set subscription
az account set --subscription "your-subscription-id"

# 3. Create resource group
az group create --name "rg-prodsegmenter-dev" --location "eastus"

# 4. Deploy infrastructure
az deployment group create \
  --resource-group "rg-prodsegmenter-dev" \
  --template-file "azure-resources.bicep" \
  --parameters "@parameters.json"
```

## ⚙️ Configuration

### Environment Parameters

| Parameter | Description | Dev | Staging | Prod |
|-----------|-------------|-----|---------|------|
| `mlComputeSku` | Azure ML compute SKU | Standard_DS3_v2 | Standard_NC6s_v3 | Standard_NC12s_v3 |
| `mlComputeMaxNodes` | Max compute nodes | 2 | 4 | 8 |
| `location` | Azure region | eastus | eastus | eastus |

### Customization

To customize for your organization:

1. **Update parameters.json** with your specific values
2. **Modify azure-resources.bicep** if you need additional resources
3. **Update deploy.sh** with your naming conventions

## 🔧 Resource Configuration

### Storage Account

- **Type**: Standard_LRS (development), Standard_GRS (production)
- **Containers**: `raw-data`, `processed-data`, `models`, `experiments`
- **Security**: Private access, HTTPS only, TLS 1.2+

### Azure ML Compute

- **Development**: CPU-only compute for testing (Standard_DS3_v2)
- **Staging**: GPU compute for validation (Standard_NC6s_v3)  
- **Production**: High-performance GPU compute (Standard_NC12s_v3)

### Function App

- **Runtime**: Python 3.10
- **Plan**: Premium (for consistent performance)
- **Features**: Always On, HTTPS only, Application Insights integration

### Security Configuration

- **Managed Identity**: System-assigned for all services
- **RBAC**: Principle of least privilege
- **Network Security**: Virtual network integration (production)
- **Secrets**: Stored in Azure Key Vault

## 📊 Monitoring & Observability

### Application Insights

- **Telemetry**: Performance, exceptions, dependencies
- **Custom Metrics**: Model accuracy, inference latency
- **Alerts**: Performance degradation, error rate spikes

### Log Analytics

- **Retention**: 30 days (development), 90 days (production)
- **Queries**: Pre-built queries for common scenarios
- **Dashboards**: Real-time monitoring dashboards

## 🔄 CI/CD Integration

The infrastructure deployment integrates with GitHub Actions:

```yaml
# In your GitHub Actions workflow
- name: Deploy Infrastructure
  run: |
    cd deployment/infrastructure
    ./deploy.sh ${{ github.event.inputs.environment }}
```

## 💰 Cost Optimization

### Development Environment
- Use CPU compute for initial development
- Scale to zero when not in use
- Standard storage tier

### Production Environment  
- Reserved instances for consistent workloads
- Premium storage for better performance
- Auto-scaling based on demand

### Cost Monitoring
- Set up budget alerts
- Use Azure Cost Management
- Regular cost reviews

## 🛠️ Troubleshooting

### Common Issues

1. **Deployment Timeout**
   ```bash
   # Increase timeout
   az config set core.cli_timeout=1800
   ```

2. **Permission Errors**
   ```bash
   # Check permissions
   az role assignment list --assignee $(az account show --query user.name -o tsv)
   ```

3. **Resource Name Conflicts**
   - Update `projectName` parameter to be unique
   - Check Azure naming conventions

### Useful Commands

```bash
# Check deployment status
az deployment group show --resource-group "rg-prodsegmenter-dev" --name "deployment-name"

# List all resources
az resource list --resource-group "rg-prodsegmenter-dev" --output table

# Get deployment outputs
az deployment group show --resource-group "rg-prodsegmenter-dev" --name "deployment-name" --query properties.outputs
```

## 📋 Post-Deployment Steps

After successful deployment:

1. **Configure local environment** with output values
2. **Upload initial data** to storage containers  
3. **Test ML workspace** connectivity
4. **Verify Function App** deployment
5. **Set up monitoring alerts**

## 🔐 Security Considerations

- **Network isolation** in production environments
- **Private endpoints** for storage and key vault
- **Azure Defender** for advanced threat protection
- **Regular security assessments**

## 📞 Support

For infrastructure issues:
- Check the [troubleshooting guide](../docs/troubleshooting.md)
- Review Azure resource logs
- Contact the DevOps team

---

*Infrastructure maintained by the ProdSegmenter DevOps team* 