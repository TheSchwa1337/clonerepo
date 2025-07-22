#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SCHWABOT DISTRIBUTED SYSTEM LAUNCHER
======================================

Simple launcher for the Schwabot distributed trading system.
Automatically detects node type and starts the appropriate service.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Schwabot Distributed System')
    parser.add_argument('--node-type', choices=['master', 'worker', 'auto'], 
                       default='auto', help='Node type to start')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--port', type=int, help='Port to use')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Determine node type
    if args.node_type == 'auto':
        # Auto-detect based on hardware
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            
            # Check for GPU
            gpu_available = False
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                gpu_available = result.returncode == 0
            except:
                pass
            
            if gpu_available and cpu_count >= 8:
                node_type = 'master'
            else:
                node_type = 'worker'
                
            print(f"Auto-detected node type: {node_type}")
        except:
            node_type = 'worker'
    else:
        node_type = args.node_type
    
    # Start the appropriate node
    if node_type == 'master':
        print("🚀 Starting Master Node...")
        os.system(f"python {Path(__file__).parent}/master_node.py")
    else:
        print("🖥️ Starting Worker Node...")
        os.system(f"python {Path(__file__).parent}/worker_node.py")

if __name__ == "__main__":
    main() 