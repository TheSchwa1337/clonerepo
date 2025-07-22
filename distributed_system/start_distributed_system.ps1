# PowerShell launcher for Schwabot Distributed System
param(
    [string]$NodeType = "auto",
    [string]$Config = "",
    [int]$Port = 0,
    [string]$Host = "0.0.0.0"
)

Write-Host "🚀 Schwabot Distributed System Launcher" -ForegroundColor Green

# Determine node type
if ($NodeType -eq "auto") {
    # Auto-detect based on hardware
    $cpuCount = (Get-WmiObject -Class Win32_Processor).NumberOfCores
    
    # Check for GPU
    $gpuAvailable = $false
    try {
        $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpu) {
            $gpuAvailable = $true
        }
    } catch {
        # GPU detection failed
    }
    
    if ($gpuAvailable -and $cpuCount -ge 8) {
        $nodeType = "master"
    } else {
        $nodeType = "worker"
    }
    
    Write-Host "Auto-detected node type: $nodeType" -ForegroundColor Yellow
} else {
    $nodeType = $NodeType
}

# Start the appropriate node
if ($nodeType -eq "master") {
    Write-Host "🚀 Starting Master Node..." -ForegroundColor Green
    python distributed_system/master_node.py
} else {
    Write-Host "🖥️ Starting Worker Node..." -ForegroundColor Green
    python distributed_system/worker_node.py
} 