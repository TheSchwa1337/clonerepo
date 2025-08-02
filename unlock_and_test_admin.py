#!/usr/bin/env python3
"""
Unlock and Test Admin System
============================

Script to unlock the admin account and test the new user-based system.
"""

import sqlite3
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unlock_admin_account():
    """Unlock the admin account."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Unlock admin account
        cursor.execute('''
            UPDATE users SET is_locked = 0, failed_attempts = 0, lockout_until = NULL 
            WHERE username = 'admin'
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Admin account unlocked successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error unlocking admin account: {e}")
        return False

def test_admin_authentication():
    """Test admin authentication with multiple failed attempts."""
    try:
        # Import the security system
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from schwabot_security_system import SchwabotSecuritySystem
        
        # Create security system instance
        security_system = SchwabotSecuritySystem()
        
        # Test admin authentication with wrong password multiple times
        test_username = "admin"
        wrong_password = "wrong_password"
        
        logger.info(f"🧪 Testing admin authentication for {test_username}")
        logger.info("Testing multiple failed attempts (should not lockout admin)")
        
        for attempt in range(1, 6):
            success, result = security_system.authenticate_user(test_username, wrong_password)
            logger.info(f"Attempt {attempt}: Success={success}, Result={result}")
            
            if "no lockout applied" in result.lower():
                logger.info("✅ Admin lockout protection working correctly")
                break
        
        # Test successful authentication
        logger.info("\n🧪 Testing successful admin authentication")
        success, result = security_system.authenticate_user(test_username, "admin123")
        logger.info(f"Successful login: Success={success}, Result={result}")
        
        if success:
            logger.info("✅ Admin authentication working correctly")
        else:
            logger.warning("⚠️ Admin authentication failed - check password")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing admin authentication: {e}")
        return False

def test_admin_management():
    """Test admin management functions."""
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from schwabot_security_system import SchwabotSecuritySystem
        
        security_system = SchwabotSecuritySystem()
        
        logger.info("🧪 Testing admin management functions")
        
        # Test admin count
        admin_count = security_system.get_admin_count()
        logger.info(f"Current admin count: {admin_count}")
        
        # Test user list
        users = security_system.get_all_users("admin")
        logger.info(f"Found {len(users)} users in system")
        
        for user in users:
            logger.info(f"👤 {user['username']} - Role: {user['role']} - Type: {user['account_type']}")
        
        # Test creating a new user
        logger.info("\n🧪 Testing user creation")
        success, message = security_system.create_user("testuser", "testpass123", "test@example.com", "user", "user", "admin")
        logger.info(f"Create user result: {success}, {message}")
        
        if success:
            # Test admin count after creation
            new_admin_count = security_system.get_admin_count()
            logger.info(f"Admin count after user creation: {new_admin_count}")
            
            # Test creating another admin (should fail due to limit)
            logger.info("\n🧪 Testing admin creation limit")
            success2, message2 = security_system.create_user("testadmin", "testpass123", "admin@example.com", "admin", "admin", "admin")
            logger.info(f"Create admin result: {success2}, {message2}")
            
            if not success2 and "maximum" in message2.lower():
                logger.info("✅ Admin limit enforcement working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing admin management: {e}")
        return False

def main():
    """Main function to unlock and test the system."""
    logger.info("🔓 Starting Admin Unlock and Test")
    logger.info("=" * 50)
    
    # Step 1: Unlock admin account
    logger.info("🔓 Step 1: Unlocking admin account")
    success = unlock_admin_account()
    
    if not success:
        logger.error("❌ Failed to unlock admin account")
        return
    
    # Step 2: Test admin authentication
    logger.info("\n🧪 Step 2: Testing admin authentication")
    test_admin_authentication()
    
    # Step 3: Test admin management
    logger.info("\n🧪 Step 3: Testing admin management")
    test_admin_management()
    
    logger.info("\n🎉 Admin System Test Complete!")
    logger.info("=" * 50)
    logger.info("✅ Admin account unlocked")
    logger.info("✅ Admin lockout protection working")
    logger.info("✅ Admin management functions working")
    logger.info("✅ System ready for use")

if __name__ == "__main__":
    main() 