#!/bin/bash
# Azure Infrastructure Deployment Script for ProdSegmenter

set -e

# Configuration
RESOURCE_GROUP_PREFIX="rg-prodsegmenter"
LOCATION="eastus"
ENVIRONMENT="${1:-dev}"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validate prerequisites
validate_prerequisites() {
    log_info "Validating prerequisites..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        log_error "Not logged in to Azure. Please run 'az login' first."
        exit 1
    fi
    
    # Check if subscription is set
    if [ -z "$SUBSCRIPTION_ID" ]; then
        log_warning "AZURE_SUBSCRIPTION_ID not set. Using current subscription."
        SUBSCRIPTION_ID=$(az account show --query id -o tsv)
    fi
    
    log_success "Prerequisites validated"
}

# Create or update resource group
create_resource_group() {
    local rg_name="${RESOURCE_GROUP_PREFIX}-${ENVIRONMENT}"
    
    log_info "Creating resource group: ${rg_name}"
    
    az group create \
        --name "$rg_name" \
        --location "$LOCATION" \
        --subscription "$SUBSCRIPTION_ID"
    
    log_success "Resource group created: ${rg_name}"
}

# Deploy infrastructure using Bicep
deploy_infrastructure() {
    local rg_name="${RESOURCE_GROUP_PREFIX}-${ENVIRONMENT}"
    local deployment_name="prodsegmenter-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S)"
    
    log_info "Deploying infrastructure..."
    log_info "Resource Group: ${rg_name}"
    log_info "Deployment Name: ${deployment_name}"
    log_info "Environment: ${ENVIRONMENT}"
    
    # Update parameters based on environment
    if [ "$ENVIRONMENT" = "prod" ]; then
        ML_COMPUTE_SKU="Standard_NC12s_v3"
        ML_COMPUTE_MAX_NODES=8
    elif [ "$ENVIRONMENT" = "staging" ]; then
        ML_COMPUTE_SKU="Standard_NC6s_v3"
        ML_COMPUTE_MAX_NODES=4
    else
        ML_COMPUTE_SKU="Standard_DS3_v2"
        ML_COMPUTE_MAX_NODES=2
    fi
    
    # Deploy using Bicep template
    deployment_output=$(az deployment group create \
        --resource-group "$rg_name" \
        --name "$deployment_name" \
        --template-file "azure-resources.bicep" \
        --parameters environmentName="$ENVIRONMENT" \
                    projectName="prodsegmenter" \
                    location="$LOCATION" \
                    mlComputeSku="$ML_COMPUTE_SKU" \
                    mlComputeMaxNodes="$ML_COMPUTE_MAX_NODES" \
        --output json)
    
    if [ $? -eq 0 ]; then
        log_success "Infrastructure deployment completed successfully"
        
        # Extract important outputs
        function_app_url=$(echo "$deployment_output" | jq -r '.properties.outputs.functionAppUrl.value')
        streamlit_app_url=$(echo "$deployment_output" | jq -r '.properties.outputs.streamlitAppUrl.value')
        ml_workspace_url=$(echo "$deployment_output" | jq -r '.properties.outputs.mlWorkspaceUrl.value')
        storage_account_name=$(echo "$deployment_output" | jq -r '.properties.outputs.storageAccountName.value')
        
        log_info "Deployment outputs:"
        echo "  Function App URL: $function_app_url"
        echo "  Streamlit App URL: $streamlit_app_url"
        echo "  ML Workspace URL: $ml_workspace_url"
        echo "  Storage Account: $storage_account_name"
        
        # Save outputs to file
        echo "$deployment_output" > "deployment-outputs-${ENVIRONMENT}.json"
        log_success "Deployment outputs saved to deployment-outputs-${ENVIRONMENT}.json"
        
    else
        log_error "Infrastructure deployment failed"
        exit 1
    fi
}

# Configure Azure ML workspace
configure_ml_workspace() {
    local rg_name="${RESOURCE_GROUP_PREFIX}-${ENVIRONMENT}"
    local workspace_name="prodsegmenter-${ENVIRONMENT}-ml"
    
    log_info "Configuring Azure ML workspace..."
    
    # Install Azure ML CLI extension if not present
    if ! az extension list | grep -q "ml"; then
        log_info "Installing Azure ML CLI extension..."
        az extension add --name ml --yes
    fi
    
    # Set default workspace
    az configure --defaults group="$rg_name" workspace="$workspace_name"
    
    log_success "Azure ML workspace configured"
}

# Create initial folder structure in storage
setup_storage_structure() {
    local storage_account_name=$(jq -r '.properties.outputs.storageAccountName.value' "deployment-outputs-${ENVIRONMENT}.json")
    
    log_info "Setting up storage folder structure..."
    
    # Create folder structure using Azure CLI
    local containers=("raw-data" "processed-data" "models" "experiments")
    
    for container in "${containers[@]}"; do
        log_info "Setting up folder structure in container: $container"
        
        # Create placeholder files to establish folder structure
        case $container in
            "raw-data")
                echo "Raw dataset storage" > temp.txt
                az storage blob upload --account-name "$storage_account_name" --container-name "$container" --name "README.md" --file temp.txt --overwrite
                ;;
            "processed-data")
                echo "Processed dataset storage" > temp.txt
                az storage blob upload --account-name "$storage_account_name" --container-name "$container" --name "README.md" --file temp.txt --overwrite
                ;;
            "models")
                echo "Model artifacts storage" > temp.txt
                az storage blob upload --account-name "$storage_account_name" --container-name "$container" --name "README.md" --file temp.txt --overwrite
                ;;
            "experiments")
                echo "Experiment tracking storage" > temp.txt
                az storage blob upload --account-name "$storage_account_name" --container-name "$container" --name "README.md" --file temp.txt --overwrite
                ;;
        esac
    done
    
    rm -f temp.txt
    log_success "Storage folder structure created"
}

# Display usage
show_usage() {
    echo "Usage: $0 [environment]"
    echo ""
    echo "Arguments:"
    echo "  environment    Target environment (dev, staging, prod). Default: dev"
    echo ""
    echo "Environment Variables:"
    echo "  AZURE_SUBSCRIPTION_ID    Azure subscription ID (optional)"
    echo ""
    echo "Examples:"
    echo "  $0              # Deploy to dev environment"
    echo "  $0 staging      # Deploy to staging environment"
    echo "  $0 prod         # Deploy to production environment"
}

# Main execution
main() {
    echo "=========================================="
    echo "   ProdSegmenter Infrastructure Deployment"
    echo "=========================================="
    echo ""
    
    # Check for help
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    fi
    
    # Validate environment parameter
    if [ -n "$1" ] && [[ ! "$1" =~ ^(dev|staging|prod)$ ]]; then
        log_error "Invalid environment: $1. Must be dev, staging, or prod."
        show_usage
        exit 1
    fi
    
    log_info "Starting deployment for environment: ${ENVIRONMENT}"
    log_info "Target location: ${LOCATION}"
    
    # Execute deployment steps
    validate_prerequisites
    create_resource_group
    deploy_infrastructure
    configure_ml_workspace
    setup_storage_structure
    
    echo ""
    echo "=========================================="
    log_success "Deployment completed successfully!"
    echo "=========================================="
    echo ""
    log_info "Next steps:"
    echo "1. Upload your training data to the storage account"
    echo "2. Configure your local environment with the deployment outputs"
    echo "3. Run the training pipeline using Azure ML"
    echo "4. Deploy your model to the Function App"
    echo ""
    log_info "For more information, check the deployment outputs file:"
    echo "  deployment-outputs-${ENVIRONMENT}.json"
}

# Execute main function
main "$@" 