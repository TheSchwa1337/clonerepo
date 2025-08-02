#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Repository Reorganization
================================

Direct file reorganization without complex verification.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create organized directory structure."""
    directories = [
        "tests/integration",
        "tests/unit", 
        "tests/performance",
        "tests/security",
        "tests/demos",
        "docs/guides",
        "docs/api",
        "docs/deployment",
        "docs/development",
        "docs/summaries",
        "monitoring/logs",
        "monitoring/reports",
        "monitoring/metrics",
        "development/scripts",
        "development/fixes",
        "development/tools",
        "config",
        "requirements"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def move_files():
    """Move files to organized locations."""
    moved_files = []
    
    # Test files
    test_files = list(Path('.').glob('test_*.py'))
    for file_path in test_files:
        try:
            destination = Path('tests/integration') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"test: {file_path.name}")
            logger.info(f"Moved test file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Demo files
    demo_files = list(Path('.').glob('demo_*.py'))
    for file_path in demo_files:
        try:
            destination = Path('tests/demos') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"demo: {file_path.name}")
            logger.info(f"Moved demo file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Documentation files
    doc_files = list(Path('.').glob('*.md'))
    for file_path in doc_files:
        if file_path.name in ['README.md']:  # Keep main README
            continue
        try:
            if 'SUMMARY' in file_path.name:
                destination = Path('docs/summaries') / file_path.name
            elif 'GUIDE' in file_path.name or 'OVERVIEW' in file_path.name:
                destination = Path('docs/guides') / file_path.name
            else:
                destination = Path('docs') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"doc: {file_path.name}")
            logger.info(f"Moved doc file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Log files
    log_files = list(Path('.').glob('*.log'))
    for file_path in log_files:
        try:
            destination = Path('monitoring/logs') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"log: {file_path.name}")
            logger.info(f"Moved log file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Report files
    report_files = list(Path('.').glob('*_report.json'))
    for file_path in report_files:
        try:
            destination = Path('monitoring/reports') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"report: {file_path.name}")
            logger.info(f"Moved report file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Fix scripts
    fix_files = list(Path('.').glob('fix_*.py'))
    for file_path in fix_files:
        try:
            destination = Path('development/fixes') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"fix: {file_path.name}")
            logger.info(f"Moved fix file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Enhanced scripts
    enhanced_files = list(Path('.').glob('enhanced_*.py'))
    for file_path in enhanced_files:
        try:
            destination = Path('development/scripts') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"enhanced: {file_path.name}")
            logger.info(f"Moved enhanced file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Config files
    config_files = list(Path('.').glob('demo_*.json'))
    for file_path in config_files:
        try:
            destination = Path('config') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"config: {file_path.name}")
            logger.info(f"Moved config file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    # Requirements files
    req_files = list(Path('.').glob('requirements*.txt'))
    for file_path in req_files:
        if file_path.name == 'requirements.txt':  # Keep main requirements
            continue
        try:
            destination = Path('requirements') / file_path.name
            shutil.move(str(file_path), str(destination))
            moved_files.append(f"requirements: {file_path.name}")
            logger.info(f"Moved requirements file: {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to move {file_path}: {e}")
    
    return moved_files

def create_readme_files():
    """Create README files for organized directories."""
    readme_content = {
        'tests': """# Tests Directory

This directory contains all test files organized by type:

- `integration/` - Integration tests
- `unit/` - Unit tests  
- `performance/` - Performance tests
- `security/` - Security tests
- `demos/` - Demo scripts

""",
        'docs': """# Documentation

This directory contains all documentation:

- `guides/` - User guides and tutorials
- `api/` - API documentation
- `deployment/` - Deployment guides
- `development/` - Developer documentation
- `summaries/` - Summary documents

""",
        'monitoring': """# Monitoring

This directory contains logs, reports, and metrics:

- `logs/` - System logs
- `reports/` - Generated reports
- `metrics/` - Performance metrics

""",
        'development': """# Development

This directory contains development tools and scripts:

- `scripts/` - Utility scripts
- `fixes/` - Fix scripts
- `tools/` - Development tools

"""
    }
    
    for directory, content in readme_content.items():
        readme_path = Path(directory) / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(content)
        logger.info(f"Created README: {readme_path}")

def main():
    """Main reorganization function."""
    print("🚀 Starting simple repository reorganization...")
    
    # Create directories
    create_directories()
    
    # Move files
    moved_files = move_files()
    
    # Create README files
    create_readme_files()
    
    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'moved_files': moved_files,
        'total_files_moved': len(moved_files)
    }
    
    with open('simple_reorganization_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Reorganization completed!")
    print(f"📊 Total files moved: {len(moved_files)}")
    print(f"📋 Report saved to: simple_reorganization_report.json")
    
    return True

if __name__ == "__main__":
    main() 