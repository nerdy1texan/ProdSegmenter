// Azure Bicep template for ProdSegmenter infrastructure
// Deploys all required Azure resources for the ML pipeline

@description('Environment name (dev, staging, prod)')
param environmentName string = 'dev'

@description('Project name prefix for all resources')
param projectName string = 'prodsegmenter'

@description('Azure region for deployment')
param location string = resourceGroup().location

@description('Azure ML compute instance SKU')
param mlComputeSku string = 'Standard_DS3_v2'

@description('Azure ML compute cluster max nodes')
param mlComputeMaxNodes int = 4

// Variables
var resourcePrefix = '${projectName}-${environmentName}'
var storageAccountName = toLower(replace('${resourcePrefix}storage', '-', ''))
var keyVaultName = '${resourcePrefix}-kv'
var appInsightsName = '${resourcePrefix}-insights'
var mlWorkspaceName = '${resourcePrefix}-ml'
var functionAppName = '${resourcePrefix}-api'
var appServicePlanName = '${resourcePrefix}-plan'
var containerRegistryName = toLower(replace('${resourcePrefix}registry', '-', ''))
var streamlitAppName = '${resourcePrefix}-frontend'

// Storage Account for data and models
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    networkAcls: {
      defaultAction: 'Allow'
    }
  }

  // Blob containers
  resource blobService 'blobServices@2023-01-01' = {
    name: 'default'
    
    resource rawDataContainer 'containers@2023-01-01' = {
      name: 'raw-data'
      properties: {
        publicAccess: 'None'
      }
    }
    
    resource processedDataContainer 'containers@2023-01-01' = {
      name: 'processed-data'
      properties: {
        publicAccess: 'None'
      }
    }
    
    resource modelsContainer 'containers@2023-01-01' = {
      name: 'models'
      properties: {
        publicAccess: 'None'
      }
    }
    
    resource experimentsContainer 'containers@2023-01-01' = {
      name: 'experiments'
      properties: {
        publicAccess: 'None'
      }
    }
  }
}

// Key Vault for secrets
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: []
    enabledForDeployment: false
    enabledForTemplateDeployment: false
    enabledForDiskEncryption: false
    enableRbacAuthorization: true
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Application Insights for monitoring
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Flow_Type: 'Redfield'
    Request_Source: 'rest'
    RetentionInDays: 90
    WorkspaceResourceId: logAnalyticsWorkspace.id
  }
}

// Log Analytics Workspace
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${resourcePrefix}-logs'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
}

// Container Registry for custom images
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: containerRegistryName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    adminUserEnabled: true
    networkRuleSet: {
      defaultAction: 'Allow'
    }
    policies: {
      retentionPolicy: {
        status: 'enabled'
        days: 30
      }
    }
  }
}

// Azure Machine Learning Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-10-01' = {
  name: mlWorkspaceName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'ProdSegmenter ML Workspace'
    description: 'Azure ML workspace for grocery product segmentation'
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    containerRegistry: containerRegistry.id
    discoveryUrl: 'https://${location}.api.azureml.ms/discovery'
    publicNetworkAccess: 'Enabled'
  }
}

// ML Compute Cluster for training
resource mlComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2023-10-01' = {
  parent: mlWorkspace
  name: 'gpu-cluster'
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: mlComputeSku
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: mlComputeMaxNodes
        nodeIdleTimeBeforeScaleDown: 'PT2M'
      }
      osType: 'Linux'
      enableNodePublicIp: false
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}

// App Service Plan for hosting
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: 'P1v3'
    tier: 'PremiumV3'
  }
  properties: {
    reserved: true
  }
  kind: 'linux'
}

// Function App for inference API
resource functionApp 'Microsoft.Web/sites@2023-01-01' = {
  name: functionAppName
  location: location
  kind: 'functionapp,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion: 'Python|3.10'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        {
          name: 'FUNCTIONS_EXTENSION_VERSION'
          value: '~4'
        }
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'AzureWebJobsStorage'
          value: 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_ML_WORKSPACE'
          value: mlWorkspace.name
        }
        {
          name: 'AZURE_STORAGE_ACCOUNT'
          value: storageAccount.name
        }
        {
          name: 'MODEL_CONTAINER_NAME'
          value: 'models'
        }
      ]
    }
    httpsOnly: true
  }
}

// Streamlit Frontend Web App
resource streamlitApp 'Microsoft.Web/sites@2023-01-01' = {
  name: streamlitAppName
  location: location
  kind: 'app,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    reserved: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.10'
      alwaysOn: true
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      appSettings: [
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: 'true'
        }
        {
          name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
          value: appInsights.properties.ConnectionString
        }
        {
          name: 'AZURE_FUNCTION_ENDPOINT'
          value: 'https://${functionApp.properties.defaultHostName}/api'
        }
        {
          name: 'ENVIRONMENT'
          value: environmentName
        }
      ]
    }
    httpsOnly: true
  }
}

// Role assignments for managed identities
resource mlWorkspaceStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, mlWorkspace.id, 'Storage Blob Data Contributor')
  properties: {
    principalId: mlWorkspace.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
    principalType: 'ServicePrincipal'
  }
}

resource functionAppStorageRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, functionApp.id, 'Storage Blob Data Reader')
  properties: {
    principalId: functionApp.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '2a2b9908-6ea1-4ae2-8e65-a410df84e7d1') // Storage Blob Data Reader
    principalType: 'ServicePrincipal'
  }
}

resource functionAppMLRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: mlWorkspace
  name: guid(mlWorkspace.id, functionApp.id, 'AzureML Data Scientist')
  properties: {
    principalId: functionApp.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'f6c7c914-8db3-469d-8ca1-694a8f32e121') // AzureML Data Scientist
    principalType: 'ServicePrincipal'
  }
}

// Outputs
output resourceGroupName string = resourceGroup().name
output storageAccountName string = storageAccount.name
output mlWorkspaceName string = mlWorkspace.name
output functionAppName string = functionApp.name
output streamlitAppName string = streamlitApp.name
output containerRegistryName string = containerRegistry.name
output appInsightsName string = appInsights.name
output keyVaultName string = keyVault.name

output functionAppUrl string = 'https://${functionApp.properties.defaultHostName}'
output streamlitAppUrl string = 'https://${streamlitApp.properties.defaultHostName}'
output mlWorkspaceUrl string = 'https://ml.azure.com/workspaces/${mlWorkspace.name}'

output storageConnectionString string = 'DefaultEndpointsProtocol=https;AccountName=${storageAccount.name};AccountKey=${storageAccount.listKeys().keys[0].value};EndpointSuffix=${environment().suffixes.storage}'
output appInsightsConnectionString string = appInsights.properties.ConnectionString 