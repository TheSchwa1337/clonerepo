#!/usr/bin/env python3
"""
Schwabot USB Memory Management System
====================================

Manages USB memory for portable Schwabot trading bot:
- Automatic USB detection
- Memory backup and restoration
- Safe shutdown/startup
- Continuous memory synchronization
"""

import os
import shutil
import json
import time
import threading
from pathlib import Path
from datetime import datetime
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('schwabot_usb_memory.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SchwabotUSBMemory:
    """USB Memory Management for Schwabot trading bot."""
    
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.absolute()
        self.usb_memory_dir = None
        self.last_backup_time = None
        self.backup_interval = 60  # seconds
        self.sync_thread = None
        self.stop_sync = False
        
        # Memory directories
        self.memory_dirs = {
            'config': 'config',
            'state': 'state',
            'logs': 'logs',
            'backups': 'backups',
            'data': 'data'
        }
        
        # Files to backup
        self.critical_files = [
            'schwabot_config.json',
            'schwabot_trading_bot.log',
            'schwabot_monitoring.log',
            'schwabot_gui.log',
            'schwabot_cli.log'
        ]
        
        # Initialize USB memory
        self.initialize_usb_memory()
    
    def initialize_usb_memory(self):
        """Initialize USB memory system."""
        try:
            # Look for USB drives
            usb_drives = self.find_usb_drives()
            
            if usb_drives:
                # Use first USB drive found
                self.usb_memory_dir = usb_drives[0] / "SchwabotMemory"
                self.usb_memory_dir.mkdir(exist_ok=True)
                logger.info(f"USB memory initialized: {self.usb_memory_dir}")
                print(f"✅ USB memory initialized: {self.usb_memory_dir}")
            else:
                # Use local memory directory
                self.usb_memory_dir = self.base_dir / "SchwabotMemory"
                self.usb_memory_dir.mkdir(exist_ok=True)
                logger.info(f"Local memory initialized: {self.usb_memory_dir}")
                print(f"ℹ️ Local memory initialized: {self.usb_memory_dir}")
            
            # Create memory subdirectories
            for dir_name in self.memory_dirs.values():
                (self.usb_memory_dir / dir_name).mkdir(exist_ok=True)
            
            # Start sync thread
            self.start_sync_thread()
            
        except Exception as e:
            logger.error(f"Error initializing USB memory: {e}")
            print(f"❌ Error initializing USB memory: {e}")
    
    def find_usb_drives(self):
        """Find available USB drives."""
        usb_drives = []
        
        try:
            if sys.platform == "win32":
                # Windows: Look for removable drives using multiple methods
                try:
                    import win32api
                    import win32file
                    import win32com.client
                    
                    # Method 1: Using WMI to get detailed drive information
                    try:
                        wmi = win32com.client.GetObject("winmgmts:")
                        drives = wmi.InstancesOf("Win32_LogicalDisk")
                        
                        for drive in drives:
                            try:
                                drive_letter = drive.DeviceID
                                drive_type = drive.DriveType
                                drive_size = drive.Size
                                
                                # DriveType 2 = Removable drive
                                if drive_type == 2 and drive_size and int(drive_size) > 0:
                                    drive_path = Path(f"{drive_letter}\\")
                                    if drive_path.exists():
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via WMI: {drive_letter} (Size: {int(drive_size)/(1024**3):.1f}GB)")
                            except Exception as e:
                                logger.debug(f"Error checking drive {drive.DeviceID}: {e}")
                                continue
                    except Exception as e:
                        logger.debug(f"WMI method failed: {e}")
                    
                    # Method 2: Using win32api as fallback
                    if not usb_drives:
                        drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
                        for drive in drives:
                            try:
                                drive_type = win32file.GetDriveType(drive)
                                if drive_type == win32file.DRIVE_REMOVABLE:
                                    drive_path = Path(drive)
                                    if drive_path.exists():
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via win32api: {drive}")
                            except Exception as e:
                                logger.debug(f"Error checking drive {drive}: {e}")
                                continue
                    
                    # Method 3: Check common USB drive letters
                    if not usb_drives:
                        common_letters = ['D:', 'E:', 'F:', 'G:', 'H:', 'I:', 'J:', 'K:', 'L:', 'M:', 'N:', 'O:', 'P:', 'Q:', 'R:', 'S:', 'T:', 'U:', 'V:', 'W:', 'X:', 'Y:', 'Z:']
                        for letter in common_letters:
                            drive_path = Path(letter)
                            if drive_path.exists():
                                try:
                                    # Check if it's actually removable
                                    drive_type = win32file.GetDriveType(letter)
                                    if drive_type == win32file.DRIVE_REMOVABLE:
                                        usb_drives.append(drive_path)
                                        logger.info(f"Found USB drive via letter check: {letter}")
                                except:
                                    # If we can't determine type, assume it might be USB
                                    usb_drives.append(drive_path)
                                    logger.info(f"Found potential USB drive: {letter}")
                    
                except ImportError:
                    logger.warning("win32api not available, using fallback method")
                    # Fallback: check common USB drive letters
                    common_letters = ['D:', 'E:', 'F:', 'G:', 'H:', 'I:', 'J:', 'K:', 'L:', 'M:', 'N:', 'O:', 'P:', 'Q:', 'R:', 'S:', 'T:', 'U:', 'V:', 'W:', 'X:', 'Y:', 'Z:']
                    for letter in common_letters:
                        drive_path = Path(letter)
                        if drive_path.exists():
                            usb_drives.append(drive_path)
                            logger.info(f"Found potential USB drive (fallback): {letter}")
            else:
                # Linux/Mac: Look for mounted USB devices
                try:
                    import subprocess
                    result = subprocess.run(['mount'], capture_output=True, text=True)
                    for line in result.stdout.split('\n'):
                        if 'usb' in line.lower() or '/media/' in line or '/mnt/' in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                usb_drives.append(Path(parts[2]))
                except:
                    # Fallback: check common mount points
                    common_mounts = ['/media', '/mnt', '/run/media']
                    for mount in common_mounts:
                        mount_path = Path(mount)
                        if mount_path.exists():
                            for item in mount_path.iterdir():
                                if item.is_dir():
                                    usb_drives.append(item)
            
        except Exception as e:
            logger.error(f"Error finding USB drives: {e}")
        
        # Remove duplicates and sort
        usb_drives = list(set(usb_drives))
        usb_drives.sort()
        
        logger.info(f"Found {len(usb_drives)} USB drives: {[str(d) for d in usb_drives]}")
        return usb_drives
    
    def start_sync_thread(self):
        """Start the memory synchronization thread."""
        self.stop_sync = False
        self.sync_thread = threading.Thread(target=self.sync_loop, daemon=True)
        self.sync_thread.start()
        logger.info("Memory sync thread started")
    
    def sync_loop(self):
        """Main synchronization loop."""
        while not self.stop_sync:
            try:
                # Check for USB drives
                usb_drives = self.find_usb_drives()
                
                if usb_drives and self.usb_memory_dir:
                    # Check if current USB is still available
                    current_usb = self.usb_memory_dir.parent
                    if current_usb not in usb_drives:
                        # USB was disconnected, switch to local
                        logger.warning("USB disconnected, switching to local memory")
                        self.usb_memory_dir = self.base_dir / "SchwabotMemory"
                        self.usb_memory_dir.mkdir(exist_ok=True)
                    else:
                        # USB is available, perform backup
                        if self.should_backup():
                            self.backup_memory_silent()
                
                # Sleep
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def should_backup(self):
        """Check if backup is needed."""
        if self.last_backup_time is None:
            return True
        
        time_since_backup = (datetime.now() - self.last_backup_time).total_seconds()
        return time_since_backup > self.backup_interval
    
    def backup_memory(self, force=False):
        """Backup memory to USB."""
        try:
            if not force and not self.should_backup():
                logger.info("Backup not needed yet")
                return True
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.usb_memory_dir / "backups" / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup critical files
            for file_name in self.critical_files:
                source_file = self.base_dir / file_name
                if source_file.exists():
                    shutil.copy2(source_file, backup_dir / file_name)
                    logger.info(f"Backed up: {file_name}")
            
            # Backup state directory
            state_dir = self.base_dir / "state"
            if state_dir.exists():
                shutil.copytree(state_dir, backup_dir / "state", dirs_exist_ok=True)
                logger.info("Backed up: state directory")
            
            # Backup config directory
            config_dir = self.base_dir / "config"
            if config_dir.exists():
                shutil.copytree(config_dir, backup_dir / "config", dirs_exist_ok=True)
                logger.info("Backed up: config directory")
            
            # Create backup metadata
            metadata = {
                'timestamp': timestamp,
                'backup_time': datetime.now().isoformat(),
                'files_backed_up': len(list(backup_dir.glob('*'))),
                'source_directory': str(self.base_dir),
                'usb_directory': str(self.usb_memory_dir)
            }
            
            with open(backup_dir / 'backup_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_backup_time = datetime.now()
            logger.info(f"Memory backup completed: {backup_dir}")
            print(f"💾 Memory backed up: {backup_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error backing up memory: {e}")
            print(f"❌ Error backing up memory: {e}")
            return False
    
    def backup_memory_silent(self):
        """Silent memory backup (no user interaction)."""
        return self.backup_memory(force=False)
    
    def restore_memory(self, backup_name=None):
        """Restore memory from USB."""
        try:
            backup_dir = self.usb_memory_dir / "backups"
            if not backup_dir.exists():
                logger.warning("No backup directory found")
                return False
            
            # Find backup to restore
            if backup_name:
                target_backup = backup_dir / backup_name
                if not target_backup.exists():
                    logger.error(f"Backup not found: {backup_name}")
                    return False
            else:
                # Use latest backup
                backups = list(backup_dir.glob("backup_*"))
                if not backups:
                    logger.warning("No backups found to restore")
                    return False
                
                target_backup = max(backups, key=lambda x: x.stat().st_mtime)
            
            logger.info(f"Restoring from backup: {target_backup}")
            print(f"🔄 Restoring from backup: {target_backup.name}")
            
            # Restore files
            restored_count = 0
            for item in target_backup.iterdir():
                if item.is_file() and item.name != 'backup_metadata.json':
                    shutil.copy2(item, self.base_dir / item.name)
                    restored_count += 1
                    logger.info(f"Restored: {item.name}")
                elif item.is_dir():
                    shutil.copytree(item, self.base_dir / item.name, dirs_exist_ok=True)
                    restored_count += 1
                    logger.info(f"Restored directory: {item.name}")
            
            logger.info(f"Memory restoration completed: {restored_count} items restored")
            print(f"✅ Memory restored: {restored_count} items")
            
            return True
            
        except Exception as e:
            logger.error(f"Error restoring memory: {e}")
            print(f"❌ Error restoring memory: {e}")
            return False
    
    def get_memory_info(self):
        """Get memory information."""
        try:
            info = {
                'usb_memory_dir': str(self.usb_memory_dir),
                'last_backup_time': self.last_backup_time.isoformat() if self.last_backup_time else None,
                'backup_interval': self.backup_interval,
                'sync_active': not self.stop_sync
            }
            
            # Get backup history
            backup_dir = self.usb_memory_dir / "backups"
            if backup_dir.exists():
                backups = list(backup_dir.glob("backup_*"))
                info['backup_count'] = len(backups)
                info['latest_backup'] = max(backups, key=lambda x: x.stat().st_mtime).name if backups else None
            else:
                info['backup_count'] = 0
                info['latest_backup'] = None
            
            # Get memory usage
            try:
                total_size = sum(f.stat().st_size for f in self.usb_memory_dir.rglob('*') if f.is_file())
                info['total_size_mb'] = total_size / (1024*1024)
            except:
                info['total_size_mb'] = 0
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return None
    
    def scan_usb(self):
        """Scan for USB drives and update memory location."""
        try:
            usb_drives = self.find_usb_drives()
            
            if usb_drives:
                # Use first USB drive found
                new_memory_dir = usb_drives[0] / "SchwabotMemory"
                new_memory_dir.mkdir(exist_ok=True)
                
                # Copy existing memory to new location
                if self.usb_memory_dir and self.usb_memory_dir.exists():
                    shutil.copytree(self.usb_memory_dir, new_memory_dir, dirs_exist_ok=True)
                
                self.usb_memory_dir = new_memory_dir
                logger.info(f"USB detected and memory moved: {usb_drives[0]}")
                print(f"🔍 USB detected: {usb_drives[0]}")
                return True
            else:
                logger.info("No USB drives found")
                print("🔍 No USB drives found")
                return False
            
        except Exception as e:
            logger.error(f"Error scanning USB: {e}")
            print(f"❌ Error scanning USB: {e}")
            return False
    
    def safe_shutdown(self):
        """Perform safe shutdown with memory backup."""
        try:
            logger.info("Starting safe shutdown procedure")
            print("🔒 Starting safe shutdown procedure...")
            
            # Final memory backup
            if self.backup_memory(force=True):
                logger.info("Safe shutdown completed successfully")
                print("✅ Safe shutdown completed with memory backup")
                return True
            else:
                logger.error("Safe shutdown failed - backup failed")
                print("❌ Safe shutdown failed - backup failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during safe shutdown: {e}")
            print(f"❌ Error during safe shutdown: {e}")
            return False
    
    def startup_restore(self):
        """Restore memory on startup."""
        try:
            logger.info("Starting memory restoration on startup")
            print("🔄 Restoring memory on startup...")
            
            # Scan for USB
            self.scan_usb()
            
            # Restore latest memory
            if self.restore_memory():
                logger.info("Startup memory restoration completed")
                print("✅ Startup memory restoration completed")
                return True
            else:
                logger.warning("No memory to restore on startup")
                print("ℹ️ No memory to restore on startup")
                return False
                
        except Exception as e:
            logger.error(f"Error during startup restore: {e}")
            print(f"❌ Error during startup restore: {e}")
            return False
    
    def cleanup_old_backups(self, keep_count=10):
        """Clean up old backups, keeping only the most recent ones."""
        try:
            backup_dir = self.usb_memory_dir / "backups"
            if not backup_dir.exists():
                return
            
            backups = list(backup_dir.glob("backup_*"))
            if len(backups) <= keep_count:
                return
            
            # Sort by modification time and remove old ones
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            old_backups = backups[keep_count:]
            
            for backup in old_backups:
                shutil.rmtree(backup)
                logger.info(f"Removed old backup: {backup.name}")
            
            logger.info(f"Cleaned up {len(old_backups)} old backups")
            print(f"🧹 Cleaned up {len(old_backups)} old backups")
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def stop(self):
        """Stop the USB memory system."""
        try:
            self.stop_sync = True
            if self.sync_thread:
                self.sync_thread.join(timeout=5)
            
            logger.info("USB memory system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping USB memory system: {e}")

def main():
    """Main function for testing USB memory system."""
    try:
        print("🚀 Schwabot USB Memory Management System")
        print("=" * 50)
        
        # Initialize USB memory
        usb_memory = SchwabotUSBMemory()
        
        # Show memory info
        info = usb_memory.get_memory_info()
        if info:
            print(f"\n📊 Memory Information:")
            print(f"   USB Directory: {info['usb_memory_dir']}")
            print(f"   Backup Count: {info['backup_count']}")
            print(f"   Latest Backup: {info['latest_backup']}")
            print(f"   Total Size: {info['total_size_mb']:.2f} MB")
            print(f"   Sync Active: {info['sync_active']}")
        
        # Test backup
        print(f"\n💾 Testing backup...")
        if usb_memory.backup_memory(force=True):
            print("✅ Backup test successful")
        else:
            print("❌ Backup test failed")
        
        # Test restore
        print(f"\n🔄 Testing restore...")
        if usb_memory.restore_memory():
            print("✅ Restore test successful")
        else:
            print("ℹ️ No memory to restore")
        
        # Cleanup
        usb_memory.cleanup_old_backups()
        
        # Stop
        usb_memory.stop()
        
        print(f"\n🎉 USB Memory Management System test completed!")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 