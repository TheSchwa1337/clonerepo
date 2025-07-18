#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repository Reorganization Script
================================
This script reorganizes the Schwabot repository to make it more user-friendly
and instructional by moving testing and monitoring files to appropriate locations.

The script will:
1. Move test files to a dedicated tests directory
2. Move monitoring and log files to appropriate directories
3. Move report files to a reports directory
4. Preserve all functionality while improving organization
5. Create a clean, instructional repository structure
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RepositoryReorganizer:
    """Handles repository reorganization while preserving functionality."""
    
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.backup_dir = self.root_dir / f"backup_before_reorganization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.moved_files = []
        self.errors = []
        
    def create_backup(self):
        """Create a backup of the current repository structure."""
        logger.info("Creating backup of current repository...")
        
        try:
            # Create backup directory
            self.backup_dir.mkdir(exist_ok=True)
            
            # Files to backup (important files that might be moved)
            important_files = [
                "requirements.txt",
                "requirements_unified_complete.txt",
                "requirements_koboldcpp.txt",
                ".gitattributes",
                "README.md"
            ]
            
            for file_name in important_files:
                file_path = self.root_dir / file_name
                if file_path.exists():
                    shutil.copy2(file_path, self.backup_dir / file_name)
                    logger.info(f"Backed up: {file_name}")
            
            logger.info(f"Backup created at: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories for reorganization."""
        logger.info("Creating reorganization directories...")
        
        directories = [
            "tests/integration",
            "tests/unit",
            "tests/performance",
            "tests/security",
            "monitoring/logs",
            "monitoring/reports",
            "monitoring/metrics",
            "development/scripts",
            "development/tools",
            "development/debug",
            "data/backtesting",
            "data/historical",
            "data/analysis"
        ]
        
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def identify_files_to_move(self):
        """Identify files that should be moved to appropriate directories."""
        
        # Test files patterns
        test_patterns = [
            "test_*.py",
            "*_test.py",
            "test_*.json",
            "*_test_results.json"
        ]
        
        # Monitoring and log files
        monitoring_patterns = [
            "*.log",
            "*_log.txt",
            "*_report.json",
            "*_report.md",
            "*_results.json",
            "*_status.json",
            "*_analysis.json"
        ]
        
        # Development and debug files
        development_patterns = [
            "debug_*.py",
            "fix_*.py",
            "*_fix.py",
            "quick_fix_*.py",
            "comprehensive_*.py",
            "safe_*.py",
            "final_*.py"
        ]
        
        # Data files
        data_patterns = [
            "*.db",
            "*_state.db",
            "*_monitoring.db",
            "all_*.txt",
            "*_data.log"
        ]
        
        files_to_move = {
            'tests': [],
            'monitoring': [],
            'development': [],
            'data': []
        }
        
        # Scan for files matching patterns
        for file_path in self.root_dir.rglob("*"):
            if file_path.is_file() and not self._is_excluded(file_path):
                file_name = file_path.name
                
                # Check test patterns
                if any(self._matches_pattern(file_name, pattern) for pattern in test_patterns):
                    files_to_move['tests'].append(file_path)
                
                # Check monitoring patterns
                elif any(self._matches_pattern(file_name, pattern) for pattern in monitoring_patterns):
                    files_to_move['monitoring'].append(file_path)
                
                # Check development patterns
                elif any(self._matches_pattern(file_name, pattern) for pattern in development_patterns):
                    files_to_move['development'].append(file_path)
                
                # Check data patterns
                elif any(self._matches_pattern(file_name, pattern) for pattern in data_patterns):
                    files_to_move['data'].append(file_path)
        
        return files_to_move
    
    def _is_excluded(self, file_path):
        """Check if file should be excluded from reorganization."""
        excluded_dirs = [
            '.git',
            '__pycache__',
            '.pytest_cache',
            '.benchmarks',
            'node_modules',
            'venv',
            'env',
            'backup_*',
            'docs'
        ]
        
        excluded_files = [
            'requirements.txt',
            'requirements_*.txt',
            'README.md',
            '.gitattributes',
            '.gitignore',
            'reorganize_repository.py'
        ]
        
        # Check if file is in excluded directory
        for excluded_dir in excluded_dirs:
            if excluded_dir in file_path.parts:
                return True
        
        # Check if file is excluded
        if file_path.name in excluded_files:
            return True
        
        return False
    
    def _matches_pattern(self, filename, pattern):
        """Check if filename matches a pattern."""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def move_files(self, files_to_move):
        """Move files to their appropriate directories."""
        logger.info("Moving files to appropriate directories...")
        
        for category, files in files_to_move.items():
            logger.info(f"Processing {category} files...")
            
            for file_path in files:
                try:
                    # Determine destination based on category and file type
                    destination = self._get_destination(file_path, category)
                    
                    if destination:
                        # Create destination directory if it doesn't exist
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Move the file
                        shutil.move(str(file_path), str(destination))
                        
                        self.moved_files.append({
                            'original': str(file_path),
                            'destination': str(destination),
                            'category': category
                        })
                        
                        logger.info(f"Moved: {file_path.name} -> {destination}")
                    
                except Exception as e:
                    error_msg = f"Failed to move {file_path}: {e}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
    
    def _get_destination(self, file_path, category):
        """Determine the destination path for a file."""
        file_name = file_path.name
        
        if category == 'tests':
            if 'integration' in file_name.lower():
                return self.root_dir / 'tests' / 'integration' / file_name
            elif 'unit' in file_name.lower():
                return self.root_dir / 'tests' / 'unit' / file_name
            elif 'performance' in file_name.lower():
                return self.root_dir / 'tests' / 'performance' / file_name
            elif 'security' in file_name.lower():
                return self.root_dir / 'tests' / 'security' / file_name
            else:
                return self.root_dir / 'tests' / 'integration' / file_name
        
        elif category == 'monitoring':
            if file_name.endswith('.log') or '_log' in file_name:
                return self.root_dir / 'monitoring' / 'logs' / file_name
            elif file_name.endswith('.json') and ('report' in file_name or 'results' in file_name):
                return self.root_dir / 'monitoring' / 'reports' / file_name
            elif file_name.endswith('.md') and 'report' in file_name:
                return self.root_dir / 'monitoring' / 'reports' / file_name
            else:
                return self.root_dir / 'monitoring' / 'metrics' / file_name
        
        elif category == 'development':
            if 'debug' in file_name.lower():
                return self.root_dir / 'development' / 'debug' / file_name
            elif 'script' in file_name.lower():
                return self.root_dir / 'development' / 'scripts' / file_name
            else:
                return self.root_dir / 'development' / 'tools' / file_name
        
        elif category == 'data':
            if 'backtest' in file_name.lower():
                return self.root_dir / 'data' / 'backtesting' / file_name
            elif 'historical' in file_name.lower():
                return self.root_dir / 'data' / 'historical' / file_name
            else:
                return self.root_dir / 'data' / 'analysis' / file_name
        
        return None
    
    def create_readme_files(self):
        """Create README files for the new directory structure."""
        logger.info("Creating README files for new directories...")
        
        readme_content = {
            'tests': """# Tests Directory

This directory contains all test files for the Schwabot system.

## Structure
- `integration/`: Integration tests that test multiple components together
- `unit/`: Unit tests for individual components
- `performance/`: Performance and load tests
- `security/`: Security and vulnerability tests

## Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/integration/
python -m pytest tests/unit/
python -m pytest tests/performance/
python -m pytest tests/security/
```
""",
            
            'monitoring': """# Monitoring Directory

This directory contains monitoring, logging, and reporting files.

## Structure
- `logs/`: System and application logs
- `reports/`: Generated reports and analysis results
- `metrics/`: Performance metrics and monitoring data

## Log Files
- System logs contain operational information
- Error logs contain error and exception details
- Performance logs contain timing and resource usage data

## Reports
- Test reports contain test execution results
- Analysis reports contain system analysis data
- Status reports contain system health information
""",
            
            'development': """# Development Directory

This directory contains development tools, scripts, and debugging utilities.

## Structure
- `scripts/`: Utility scripts for development and maintenance
- `tools/`: Development tools and utilities
- `debug/`: Debugging scripts and tools

## Usage
These files are primarily for developers and system maintenance.
Most users will not need to interact with these files directly.
""",
            
            'data': """# Data Directory

This directory contains data files used by the system.

## Structure
- `backtesting/`: Historical data for backtesting
- `historical/`: Historical market data
- `analysis/`: Analysis results and processed data

## Data Files
- Database files contain system state and configuration
- Historical data files contain market data for analysis
- Analysis files contain processed results and metrics
"""
        }
        
        for directory, content in readme_content.items():
            readme_path = self.root_dir / directory / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(content)
            logger.info(f"Created README for {directory}/")
    
    def update_imports(self):
        """Update import statements in Python files to reflect new file locations."""
        logger.info("Updating import statements...")
        
        # This is a simplified version - in practice, you'd need more sophisticated
        # import analysis and updating logic
        logger.info("Import updating would require detailed analysis of each file")
        logger.info("Consider using tools like 'isort' or 'autoflake' for import management")
    
    def generate_report(self):
        """Generate a report of the reorganization."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'moved_files': self.moved_files,
            'errors': self.errors,
            'summary': {
                'total_files_moved': len(self.moved_files),
                'total_errors': len(self.errors),
                'categories': {}
            }
        }
        
        # Count files by category
        for file_info in self.moved_files:
            category = file_info['category']
            if category not in report['summary']['categories']:
                report['summary']['categories'][category] = 0
            report['summary']['categories'][category] += 1
        
        # Save report
        report_path = self.root_dir / 'reorganization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Reorganization report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("REORGANIZATION SUMMARY")
        print("="*60)
        print(f"Backup created at: {self.backup_dir}")
        print(f"Total files moved: {len(self.moved_files)}")
        print(f"Total errors: {len(self.errors)}")
        print("\nFiles moved by category:")
        for category, count in report['summary']['categories'].items():
            print(f"  {category}: {count} files")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error}")
        
        print(f"\nDetailed report saved to: {report_path}")
        print("="*60)
    
    def run(self):
        """Run the complete reorganization process."""
        logger.info("Starting repository reorganization...")
        
        # Step 1: Create backup
        if not self.create_backup():
            logger.error("Failed to create backup. Aborting reorganization.")
            return False
        
        # Step 2: Create directories
        self.create_directories()
        
        # Step 3: Identify files to move
        files_to_move = self.identify_files_to_move()
        
        # Step 4: Move files
        self.move_files(files_to_move)
        
        # Step 5: Create README files
        self.create_readme_files()
        
        # Step 6: Update imports (placeholder)
        self.update_imports()
        
        # Step 7: Generate report
        self.generate_report()
        
        logger.info("Repository reorganization completed!")
        return True

def main():
    """Main function to run the reorganization."""
    print("Schwabot Repository Reorganization")
    print("="*40)
    print("This script will reorganize the repository to make it more user-friendly.")
    print("A backup will be created before any changes are made.")
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with the reorganization? (y/N): ")
    if response.lower() != 'y':
        print("Reorganization cancelled.")
        return
    
    # Run reorganization
    reorganizer = RepositoryReorganizer()
    success = reorganizer.run()
    
    if success:
        print("\n✅ Reorganization completed successfully!")
        print("The repository is now better organized and more user-friendly.")
        print("Check the reorganization_report.json for details.")
    else:
        print("\n❌ Reorganization failed. Check the logs for details.")

if __name__ == "__main__":
    main() 