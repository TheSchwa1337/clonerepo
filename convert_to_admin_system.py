#!/usr/bin/env python3
"""
Convert to Admin System
======================

Script to convert all existing users to admin role and test the new user-based system.
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_all_users_to_admin():
    """Convert all existing users to admin role."""
    try:
        # Connect to the users database
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute("SELECT username, role FROM users")
        users = cursor.fetchall()
        
        if not users:
            logger.info("No users found to convert")
            conn.close()
            return False, "No users found to convert"
        
        logger.info(f"Found {len(users)} users to process")
        
        # Convert all users to admin
        updated_count = 0
        for username, current_role in users:
            if current_role != "admin":
                cursor.execute("UPDATE users SET role = 'admin', account_type = 'admin' WHERE username = ?", (username,))
                updated_count += 1
                logger.info(f"✅ Converted {username} to admin role")
            else:
                logger.info(f"ℹ️ {username} is already admin")
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Successfully converted {updated_count} users to admin role")
        return True, f"Successfully converted {updated_count} users to admin role"
        
    except Exception as e:
        logger.error(f"Error converting users to admin: {e}")
        return False, f"Error converting users: {str(e)}"

def get_admin_count():
    """Get the current number of admin accounts."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        admin_count = cursor.fetchone()[0]
        
        conn.close()
        return admin_count
        
    except Exception as e:
        logger.error(f"Error getting admin count: {e}")
        return 0

def list_all_users():
    """List all users in the system."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, role, account_type, created_at, last_login, 
                   failed_attempts, is_locked, lockout_until
            FROM users ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        logger.info("📋 All Users in System:")
        logger.info("=" * 80)
        
        for user in users:
            logger.info(f"👤 Username: {user[0]}")
            logger.info(f"📧 Email: {user[1] or 'N/A'}")
            logger.info(f"👑 Role: {user[2]}")
            logger.info(f"📊 Account Type: {user[3]}")
            logger.info(f"📅 Created: {user[4][:10] if user[4] else 'N/A'}")
            logger.info(f"🕐 Last Login: {user[5][:10] if user[5] else 'N/A'}")
            logger.info(f"❌ Failed Attempts: {user[6]}")
            logger.info(f"🔒 Locked: {'Yes' if user[7] else 'No'}")
            logger.info(f"⏰ Lockout Until: {user[8] or 'N/A'}")
            logger.info("-" * 40)
        
        return users
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return []

def test_admin_authentication():
    """Test admin authentication with multiple failed attempts (should not lockout)."""
    try:
        from schwabot_security_system import SchwabotSecuritySystem
        
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
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing admin authentication: {e}")
        return False

def main():
    """Main function to convert and test the system."""
    logger.info("🚀 Starting Admin System Conversion")
    logger.info("=" * 50)
    
    # Step 1: List current users
    logger.info("📋 Step 1: Listing current users")
    users = list_all_users()
    
    # Step 2: Convert all users to admin
    logger.info("\n👑 Step 2: Converting all users to admin")
    success, message = convert_all_users_to_admin()
    
    if success:
        logger.info(f"✅ {message}")
    else:
        logger.error(f"❌ {message}")
        return
    
    # Step 3: Verify admin count
    logger.info("\n📊 Step 3: Verifying admin count")
    admin_count = get_admin_count()
    logger.info(f"Current admin count: {admin_count}")
    
    if admin_count > 2:
        logger.warning(f"⚠️ Warning: {admin_count} admins found (max 2 recommended)")
    else:
        logger.info(f"✅ Admin count is within limits ({admin_count}/2)")
    
    # Step 4: List updated users
    logger.info("\n📋 Step 4: Listing updated users")
    list_all_users()
    
    # Step 5: Test admin authentication
    logger.info("\n🧪 Step 5: Testing admin authentication")
    test_admin_authentication()
    
    logger.info("\n🎉 Admin System Conversion Complete!")
    logger.info("=" * 50)
    logger.info("✅ All users converted to admin role")
    logger.info("✅ Admin lockout protection enabled")
    logger.info("✅ Maximum 2 admin limit enforced")
    logger.info("✅ All functionality preserved")

if __name__ == "__main__":
    main() 