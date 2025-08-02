#!/usr/bin/env python3
"""
Reset Admin Password
==================

Script to reset the admin password to a known value.
"""

import sqlite3
import hashlib
import secrets
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reset_admin_password():
    """Reset admin password to 'admin123'."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Generate new password hash
        password = "admin123"
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        
        # Update admin password
        cursor.execute('''
            UPDATE users SET password_hash = ?, salt = ?, failed_attempts = 0, 
                           is_locked = 0, lockout_until = NULL 
            WHERE username = 'admin'
        ''', (password_hash, salt))
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Admin password reset successfully")
        logger.info(f"🔑 New password: {password}")
        logger.info(f"🧂 New salt: {salt}")
        logger.info(f"🔐 New hash: {password_hash}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error resetting admin password: {e}")
        return False

def test_new_password():
    """Test the new password."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Get admin data
        cursor.execute("SELECT password_hash, salt FROM users WHERE username = 'admin'")
        result = cursor.fetchone()
        
        if not result:
            logger.error("Admin user not found")
            return False
        
        stored_hash, salt = result
        
        # Test password
        password = "admin123"
        computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        
        if computed_hash == stored_hash:
            logger.info("✅ Password verification successful")
            return True
        else:
            logger.error("❌ Password verification failed")
            logger.error(f"Expected: {stored_hash}")
            logger.error(f"Got: {computed_hash}")
            return False
        
    except Exception as e:
        logger.error(f"Error testing password: {e}")
        return False

def main():
    """Main function."""
    logger.info("🔑 Admin Password Reset")
    logger.info("=" * 30)
    
    # Reset password
    logger.info("🔑 Resetting admin password")
    success = reset_admin_password()
    
    if not success:
        logger.error("❌ Failed to reset password")
        return
    
    # Test new password
    logger.info("\n🧪 Testing new password")
    success = test_new_password()
    
    if success:
        logger.info("✅ Password reset and test successful")
    else:
        logger.error("❌ Password test failed")
    
    logger.info("\n🎉 Password Reset Complete!")

if __name__ == "__main__":
    main() 