#!/usr/bin/env python3
"""
Simple Authentication Test
=========================

Simple test to verify admin authentication works.
"""

import sqlite3
import hashlib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_auth():
    """Test simple authentication without security system."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Test admin authentication
        username = "admin"
        password = "admin123"
        
        # Get user data
        cursor.execute("SELECT password_hash, salt, role, is_locked FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        
        if not result:
            logger.error("Admin user not found")
            return False
        
        stored_hash, salt, role, is_locked = result
        
        logger.info(f"Found user: {username}")
        logger.info(f"Role: {role}")
        logger.info(f"Locked: {is_locked}")
        
        # Verify password
        computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
        
        if computed_hash == stored_hash:
            logger.info("✅ Password verification successful")
            return True
        else:
            logger.error("❌ Password verification failed")
            return False
        
    except Exception as e:
        logger.error(f"Error in simple auth test: {e}")
        return False

def test_admin_status():
    """Test admin status and permissions."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Get all users
        cursor.execute('''
            SELECT username, role, account_type, is_locked, failed_attempts
            FROM users ORDER BY username
        ''')
        
        users = cursor.fetchall()
        
        logger.info("📋 Current Users:")
        logger.info("=" * 50)
        
        for user in users:
            logger.info(f"👤 {user[0]}")
            logger.info(f"👑 Role: {user[1]}")
            logger.info(f"📊 Type: {user[2]}")
            logger.info(f"🔒 Locked: {'Yes' if user[3] else 'No'}")
            logger.info(f"❌ Failed: {user[4]}")
            logger.info("-" * 30)
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error testing admin status: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🧪 Simple Authentication Test")
    logger.info("=" * 40)
    
    # Test admin status
    logger.info("📋 Testing admin status")
    test_admin_status()
    
    # Test simple auth
    logger.info("\n🔑 Testing simple authentication")
    success = test_simple_auth()
    
    if success:
        logger.info("✅ Authentication test passed")
    else:
        logger.error("❌ Authentication test failed")
    
    logger.info("\n🎉 Test Complete!")

if __name__ == "__main__":
    main() 