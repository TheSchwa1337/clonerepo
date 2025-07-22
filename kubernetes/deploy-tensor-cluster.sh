#!/bin/bash

# 🧮 Schwabot Tensor Cluster Deployment Script
# ============================================
# This script deploys the complete Schwabot tensor and quantum calculation
# system to Kubernetes, making tensor operations easier to manage.

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="schwabot-tensor"
CLUSTER_NAME="schwabot-tensor-cluster"
DOCKER_IMAGE="schwabot:latest"
STORAGE_CLASS="fast-ssd"

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if Docker is available (for building images)
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. You may need to build images separately."
    fi
    
    print_success "Prerequisites check completed"
}

create_namespace() {
    print_status "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        print_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace $NAMESPACE
        print_success "Namespace $NAMESPACE created"
    fi
}

build_docker_image() {
    print_status "Building Docker image: $DOCKER_IMAGE"
    
    if command -v docker &> /dev/null; then
        # Build the Schwabot image with tensor support
        docker build -t $DOCKER_IMAGE \
            --build-arg SCHWABOT_TENSOR_MODE=distributed \
            --build-arg SCHWABOT_QUANTUM_DIMENSION=16 \
            --build-arg SCHWABOT_USE_GPU=true \
            --build-arg SCHWABOT_CLUSTER_MODE=kubernetes \
            -f Dockerfile .
        
        print_success "Docker image built successfully"
    else
        print_warning "Docker not available, skipping image build"
    fi
}

deploy_custom_resources() {
    print_status "Deploying custom resource definitions..."
    
    # Deploy CRDs
    kubectl apply -f tensor-operator.yaml
    
    # Wait for CRDs to be established
    print_status "Waiting for custom resource definitions to be established..."
    kubectl wait --for=condition=established --timeout=60s crd/tensoroperations.schwabot.ai
    kubectl wait --for=condition=established --timeout=60s crd/quantumcalculations.schwabot.ai
    kubectl wait --for=condition=established --timeout=60s crd/crosschaincalculations.schwabot.ai
    
    print_success "Custom resource definitions deployed"
}

deploy_tensor_cluster() {
    print_status "Deploying Schwabot tensor cluster..."
    
    # Deploy the main cluster configuration
    kubectl apply -f schwabot-tensor-cluster.yaml
    
    # Wait for deployments to be ready
    print_status "Waiting for tensor processor deployments..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-tensor-processor -n $NAMESPACE
    
    print_status "Waiting for quantum calculator statefulsets..."
    kubectl wait --for=condition=ready --timeout=300s statefulset/schwabot-quantum-calculator -n $NAMESPACE
    
    print_status "Waiting for algebraic processor deployments..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-algebraic-processor -n $NAMESPACE
    
    print_status "Waiting for cross-chain coordinator..."
    kubectl wait --for=condition=available --timeout=300s deployment/schwabot-cross-chain-coordinator -n $NAMESPACE
    
    print_success "Tensor cluster deployed successfully"
}

deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    # Check if Prometheus Operator is installed
    if kubectl get crd prometheuses.monitoring.coreos.com &> /dev/null; then
        print_status "Prometheus Operator detected, deploying monitoring..."
        
        # Deploy Prometheus and Grafana
        kubectl apply -f monitoring/prometheus.yaml -n $NAMESPACE
        kubectl apply -f monitoring/grafana.yaml -n $NAMESPACE
        
        print_success "Monitoring stack deployed"
    else
        print_warning "Prometheus Operator not detected. Install it for full monitoring capabilities."
    fi
}

deploy_examples() {
    print_status "Deploying example tensor operations..."
    
    # Deploy example custom resources
    kubectl apply -f examples/tensor-operation-example.yaml
    
    print_success "Example tensor operations deployed"
}

validate_deployment() {
    print_status "Validating deployment..."
    
    # Check if all pods are running
    local running_pods=$(kubectl get pods -n $NAMESPACE --field-selector=status.phase=Running --no-headers | wc -l)
    local total_pods=$(kubectl get pods -n $NAMESPACE --no-headers | wc -l)
    
    if [ "$running_pods" -eq "$total_pods" ]; then
        print_success "All pods are running ($running_pods/$total_pods)"
    else
        print_warning "Some pods are not running ($running_pods/$total_pods)"
        kubectl get pods -n $NAMESPACE
    fi
    
    # Check services
    print_status "Checking services..."
    kubectl get services -n $NAMESPACE
    
    # Check custom resources
    print_status "Checking custom resources..."
    kubectl get tensoroperations -n $NAMESPACE
    kubectl get quantumcalculations -n $NAMESPACE
    kubectl get crosschaincalculations -n $NAMESPACE
    
    print_success "Deployment validation completed"
}

show_access_info() {
    print_status "Deployment completed! Here's how to access your tensor cluster:"
    echo
    echo -e "${GREEN}📊 Tensor Operations Dashboard:${NC}"
    echo "  kubectl port-forward svc/schwabot-tensor-service 8080:8080 -n $NAMESPACE"
    echo "  Then visit: http://localhost:8080"
    echo
    echo -e "${GREEN}⚛️ Quantum Calculations API:${NC}"
    echo "  kubectl port-forward svc/schwabot-quantum-service 8083:8083 -n $NAMESPACE"
    echo "  API endpoint: http://localhost:8083"
    echo
    echo -e "${GREEN}🧮 Algebraic Logic API:${NC}"
    echo "  kubectl port-forward svc/schwabot-algebraic-service 8084:8084 -n $NAMESPACE"
    echo "  API endpoint: http://localhost:8084"
    echo
    echo -e "${GREEN}🔗 Cross-Chain Coordinator API:${NC}"
    echo "  kubectl port-forward svc/schwabot-coordinator-service 8085:8085 -n $NAMESPACE"
    echo "  API endpoint: http://localhost:8085"
    echo
    echo -e "${GREEN}🎛️ Tensor Controller API:${NC}"
    echo "  kubectl port-forward svc/schwabot-tensor-controller-service 8086:8086 -n $NAMESPACE"
    echo "  API endpoint: http://localhost:8086"
    echo
    echo -e "${GREEN}📈 Monitoring (if Prometheus Operator is installed):${NC}"
    echo "  kubectl port-forward svc/prometheus-operated 9090:9090 -n $NAMESPACE"
    echo "  Prometheus: http://localhost:9090"
    echo "  kubectl port-forward svc/grafana 3000:3000 -n $NAMESPACE"
    echo "  Grafana: http://localhost:3000 (admin/admin)"
    echo
    echo -e "${GREEN}🔍 Useful Commands:${NC}"
    echo "  # View all tensor operations:"
    echo "  kubectl get tensoroperations -n $NAMESPACE"
    echo
    echo "  # View quantum calculations:"
    echo "  kubectl get quantumcalculations -n $NAMESPACE"
    echo
    echo "  # View cross-chain calculations:"
    echo "  kubectl get crosschaincalculations -n $NAMESPACE"
    echo
    echo "  # View pod logs:"
    echo "  kubectl logs -f deployment/schwabot-tensor-processor -n $NAMESPACE"
    echo
    echo "  # Scale tensor processors:"
    echo "  kubectl scale deployment schwabot-tensor-processor --replicas=5 -n $NAMESPACE"
}

create_tensor_operation() {
    print_status "Creating a test tensor operation..."
    
    cat <<EOF | kubectl apply -f -
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
EOF
    
    print_success "Test tensor operation created"
}

# Main deployment function
deploy_tensor_system() {
    print_status "🚀 Starting Schwabot Tensor Cluster Deployment"
    echo "=================================================="
    
    check_prerequisites
    create_namespace
    build_docker_image
    deploy_custom_resources
    deploy_tensor_cluster
    deploy_monitoring
    deploy_examples
    validate_deployment
    create_tensor_operation
    show_access_info
    
    print_success "🎉 Schwabot Tensor Cluster deployment completed successfully!"
    echo
    print_status "Your tensor operations are now running on Kubernetes!"
    print_status "This makes tensor operations much easier to manage with:"
    echo "  ✅ Automatic scaling based on load"
    echo "  ✅ GPU resource management"
    echo "  ✅ Distributed processing across nodes"
    echo "  ✅ Built-in monitoring and alerting"
    echo "  ✅ Custom resource definitions for tensor operations"
    echo "  ✅ Cross-chain calculation coordination"
    echo "  ✅ Quantum calculation state management"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up Schwabot tensor cluster..."
    
    # Delete custom resources
    kubectl delete -f examples/tensor-operation-example.yaml --ignore-not-found=true
    
    # Delete main cluster
    kubectl delete -f schwabot-tensor-cluster.yaml --ignore-not-found=true
    
    # Delete custom resource definitions
    kubectl delete -f tensor-operator.yaml --ignore-not-found=true
    
    # Delete namespace
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    
    print_success "Cleanup completed"
}

# Help function
show_help() {
    echo "Schwabot Tensor Cluster Deployment Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  deploy    Deploy the complete tensor cluster (default)"
    echo "  cleanup   Remove the tensor cluster"
    echo "  validate  Validate the current deployment"
    echo "  help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0 deploy    # Deploy the tensor cluster"
    echo "  $0 cleanup   # Remove the tensor cluster"
    echo "  $0 validate  # Check deployment status"
}

# Main script logic
case "${1:-deploy}" in
    deploy)
        deploy_tensor_system
        ;;
    cleanup)
        cleanup
        ;;
    validate)
        validate_deployment
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 