# 🧮 Schwabot Tensor Cluster Deployment Script (PowerShell)
# ========================================================
# This script deploys the complete Schwabot tensor and quantum calculation
# system to Kubernetes, making tensor operations easier to manage.

param(
    [Parameter(Position=0)]
    [ValidateSet("deploy", "cleanup", "validate", "help")]
    [string]$Command = "deploy"
)

# Configuration
$NAMESPACE = "schwabot-tensor"
$CLUSTER_NAME = "schwabot-tensor-cluster"
$DOCKER_IMAGE = "schwabot:latest"
$STORAGE_CLASS = "fast-ssd"

# Functions
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Test-Prerequisites {
    Write-Status "Checking prerequisites..."
    
    # Check if kubectl is installed
    try {
        $null = Get-Command kubectl -ErrorAction Stop
    }
    catch {
        Write-Error "kubectl is not installed. Please install kubectl first."
        exit 1
    }
    
    # Check if kubectl can connect to cluster
    try {
        $null = kubectl cluster-info 2>$null
    }
    catch {
        Write-Error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    }
    
    # Check if Docker is available (for building images)
    try {
        $null = Get-Command docker -ErrorAction Stop
    }
    catch {
        Write-Warning "Docker is not installed. You may need to build images separately."
    }
    
    Write-Success "Prerequisites check completed"
}

function New-Namespace {
    Write-Status "Creating namespace: $NAMESPACE"
    
    $existingNamespace = kubectl get namespace $NAMESPACE 2>$null
    if ($existingNamespace) {
        Write-Warning "Namespace $NAMESPACE already exists"
    }
    else {
        kubectl create namespace $NAMESPACE
        Write-Success "Namespace $NAMESPACE created"
    }
}

function Build-DockerImage {
    Write-Status "Building Docker image: $DOCKER_IMAGE"
    
    try {
        $null = Get-Command docker -ErrorAction Stop
        
        # Build the Schwabot image with tensor support
        docker build -t $DOCKER_IMAGE `
            --build-arg SCHWABOT_TENSOR_MODE=distributed `
            --build-arg SCHWABOT_QUANTUM_DIMENSION=16 `
            --build-arg SCHWABOT_USE_GPU=true `
            --build-arg SCHWABOT_CLUSTER_MODE=kubernetes `
            -f Dockerfile .
        
        Write-Success "Docker image built successfully"
    }
    catch {
        Write-Warning "Docker not available, skipping image build"
    }
}

function Deploy-CustomResources {
    Write-Status "Deploying custom resource definitions..."
    
    # Deploy CRDs
    kubectl apply -f tensor-operator.yaml
    
    # Wait for CRDs to be established
    Write-Status "Waiting for custom resource definitions to be established..."
    kubectl wait --for=condition=established --timeout=60s crd/tensoroperations.schwabot.ai
    kubectl wait --for=condition=established --timeout=60s crd/quantumcalculations.schwabot.ai
    kubectl wait --for=condition=established --timeout=60s crd/crosschaincalculations.schwabot.ai
    
    Write-Success "Custom resource definitions deployed"
}

function Deploy-TensorCluster {
    Write-Status "Deploying Schwabot tensor cluster..."
    
    # Deploy the main cluster configuration
    kubectl apply -f schwabot-tensor-cluster.yaml
    
    # Wait for deployments to be ready
    Write-Status "Waiting for tensor processor deployments..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-tensor-processor -n $NAMESPACE
    
    Write-Status "Waiting for quantum calculator statefulsets..."
    kubectl wait --for=condition=ready --timeout=300s statefulset/schwabot-quantum-calculator -n $NAMESPACE
    
    Write-Status "Waiting for algebraic processor deployments..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-algebraic-processor -n $NAMESPACE
    
    Write-Status "Waiting for cross-chain coordinator..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-cross-chain-coordinator -n $NAMESPACE
    
    Write-Success "Tensor cluster deployed successfully"
}

function Deploy-Monitoring {
    Write-Status "Deploying monitoring stack..."
    
    # Check if Prometheus Operator is installed
    $prometheusCRD = kubectl get crd prometheuses.monitoring.coreos.com 2>$null
    if ($prometheusCRD) {
        Write-Status "Prometheus Operator detected, deploying monitoring..."
        
        # Deploy Prometheus and Grafana (if files exist)
        if (Test-Path "monitoring/prometheus.yaml") {
            kubectl apply -f monitoring/prometheus.yaml -n $NAMESPACE
        }
        if (Test-Path "monitoring/grafana.yaml") {
            kubectl apply -f monitoring/grafana.yaml -n $NAMESPACE
        }
        
        Write-Success "Monitoring stack deployed"
    }
    else {
        Write-Warning "Prometheus Operator not detected. Install it for full monitoring capabilities."
    }
}

function Deploy-Examples {
    Write-Status "Deploying example tensor operations..."
    
    # Deploy example custom resources
    kubectl apply -f examples/tensor-operation-example.yaml
    
    Write-Success "Example tensor operations deployed"
}

function Test-Deployment {
    Write-Status "Validating deployment..."
    
    # Check if all pods are running
    $runningPods = (kubectl get pods -n $NAMESPACE --field-selector=status.phase=Running --no-headers 2>$null | Measure-Object).Count
    $totalPods = (kubectl get pods -n $NAMESPACE --no-headers 2>$null | Measure-Object).Count
    
    if ($runningPods -eq $totalPods) {
        Write-Success "All pods are running ($runningPods/$totalPods)"
    }
    else {
        Write-Warning "Some pods are not running ($runningPods/$totalPods)"
        kubectl get pods -n $NAMESPACE
    }
    
    # Check services
    Write-Status "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Check custom resources
    Write-Status "Checking custom resources..."
    kubectl get tensoroperations -n $NAMESPACE 2>$null
    kubectl get quantumcalculations -n $NAMESPACE 2>$null
    kubectl get crosschaincalculations -n $NAMESPACE 2>$null
    
    Write-Success "Deployment validation completed"
}

function Show-AccessInfo {
    Write-Status "Deployment completed! Here's how to access your tensor cluster:"
    Write-Host ""
    Write-Host "📊 Tensor Operations Dashboard:" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/schwabot-tensor-service 8080:8080 -n $NAMESPACE"
    Write-Host "  Then visit: http://localhost:8080"
    Write-Host ""
    Write-Host "⚛️ Quantum Calculations API:" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/schwabot-quantum-service 8083:8083 -n $NAMESPACE"
    Write-Host "  API endpoint: http://localhost:8083"
    Write-Host ""
    Write-Host "🧮 Algebraic Logic API:" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/schwabot-algebraic-service 8084:8084 -n $NAMESPACE"
    Write-Host "  API endpoint: http://localhost:8084"
    Write-Host ""
    Write-Host "🔗 Cross-Chain Coordinator API:" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/schwabot-coordinator-service 8085:8085 -n $NAMESPACE"
    Write-Host "  API endpoint: http://localhost:8085"
    Write-Host ""
    Write-Host "🎛️ Tensor Controller API:" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/schwabot-tensor-controller-service 8086:8086 -n $NAMESPACE"
    Write-Host "  API endpoint: http://localhost:8086"
    Write-Host ""
    Write-Host "📈 Monitoring (if Prometheus Operator is installed):" -ForegroundColor Green
    Write-Host "  kubectl port-forward svc/prometheus-operated 9090:9090 -n $NAMESPACE"
    Write-Host "  Prometheus: http://localhost:9090"
    Write-Host "  kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
    Write-Host "  Grafana: http://localhost:3000 (admin/admin)"
    Write-Host ""
    Write-Host "🔍 Useful Commands:" -ForegroundColor Green
    Write-Host "  # View all tensor operations:"
    Write-Host "  kubectl get tensoroperations -n $NAMESPACE"
    Write-Host ""
    Write-Host "  # View quantum calculations:"
    Write-Host "  kubectl get quantumcalculations -n $NAMESPACE"
    Write-Host ""
    Write-Host "  # View cross-chain calculations:"
    Write-Host "  kubectl get crosschaincalculations -n $NAMESPACE"
    Write-Host ""
    Write-Host "  # View pod logs:"
    Write-Host "  kubectl logs -f deployment/schwabot-tensor-processor -n $NAMESPACE"
    Write-Host ""
    Write-Host "  # Scale tensor processors:"
    Write-Host "  kubectl scale deployment schwabot-tensor-processor --replicas=5 -n $NAMESPACE"
}

function New-TensorOperation {
    Write-Status "Creating a test tensor operation..."
    
    $testOperation = @"
apiVersion: schwabot.ai/v1alpha1
kind: TensorOperation
metadata:
  name: test-tensor-fusion
  namespace: $NAMESPACE
spec:
  operationType: tensor_fusion
  priority: high
  tensorDepth: 4
  quantumDimension: 16
  gpuRequired: true
  memoryRequest: "1Gi"
  cpuRequest: "500m"
  timeout: "60s"
  inputData:
    tensor_a:
      shape: [32, 32]
      dtype: "float32"
      data_source: "test_data"
    tensor_b:
      shape: [32, 32]
      dtype: "float32"
      data_source: "test_data"
  expectedOutput:
    shape: [32, 32, 32, 32]
    dtype: "complex64"
    operation: "tensor_dot_fusion"
"@
    
    $testOperation | kubectl apply -f -
    
    Write-Success "Test tensor operation created"
}

function Deploy-TensorSystem {
    Write-Status "🚀 Starting Schwabot Tensor Cluster Deployment"
    Write-Host "=================================================="
    
    Test-Prerequisites
    New-Namespace
    Build-DockerImage
    Deploy-CustomResources
    Deploy-TensorCluster
    Deploy-Monitoring
    Deploy-Examples
    Test-Deployment
    New-TensorOperation
    Show-AccessInfo
    
    Write-Success "🎉 Schwabot Tensor Cluster deployment completed successfully!"
    Write-Host ""
    Write-Status "Your tensor operations are now running on Kubernetes!"
    Write-Status "This makes tensor operations much easier to manage with:"
    Write-Host "  ✅ Automatic scaling based on load"
    Write-Host "  ✅ GPU resource management"
    Write-Host "  ✅ Distributed processing across nodes"
    Write-Host "  ✅ Built-in monitoring and alerting"
    Write-Host "  ✅ Custom resource definitions for tensor operations"
    Write-Host "  ✅ Cross-chain calculation coordination"
    Write-Host "  ✅ Quantum calculation state management"
}

function Remove-TensorCluster {
    Write-Status "Cleaning up Schwabot tensor cluster..."
    
    # Delete custom resources
    kubectl delete -f examples/tensor-operation-example.yaml --ignore-not-found=true 2>$null
    
    # Delete main cluster
    kubectl delete -f schwabot-tensor-cluster.yaml --ignore-not-found=true 2>$null
    
    # Delete custom resource definitions
    kubectl delete -f tensor-operator.yaml --ignore-not-found=true 2>$null
    
    # Delete namespace
    kubectl delete namespace $NAMESPACE --ignore-not-found=true 2>$null
    
    Write-Success "Cleanup completed"
}

function Show-Help {
    Write-Host "Schwabot Tensor Cluster Deployment Script (PowerShell)"
    Write-Host ""
    Write-Host "Usage: .\deploy-tensor-cluster.ps1 [COMMAND]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  deploy    Deploy the complete tensor cluster (default)"
    Write-Host "  cleanup   Remove the tensor cluster"
    Write-Host "  validate  Validate the current deployment"
    Write-Host "  help      Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy-tensor-cluster.ps1 deploy    # Deploy the tensor cluster"
    Write-Host "  .\deploy-tensor-cluster.ps1 cleanup   # Remove the tensor cluster"
    Write-Host "  .\deploy-tensor-cluster.ps1 validate  # Check deployment status"
}

# Main script logic
switch ($Command) {
    "deploy" {
        Deploy-TensorSystem
    }
    "cleanup" {
        Remove-TensorCluster
    }
    "validate" {
        Test-Deployment
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
} 