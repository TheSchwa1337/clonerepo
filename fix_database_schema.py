#!/usr/bin/env python3
"""
Fix Database Schema
==================

Script to fix the database schema for the new user-based system.
"""

import sqlite3
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_users_database():
    """Fix the users database schema."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        # Check current schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        logger.info(f"Current columns: {column_names}")
        
        # Add missing columns if they don't exist
        if 'account_type' not in column_names:
            logger.info("Adding account_type column")
            cursor.execute("ALTER TABLE users ADD COLUMN account_type TEXT DEFAULT 'user'")
        
        if 'created_by' not in column_names:
            logger.info("Adding created_by column")
            cursor.execute("ALTER TABLE users ADD COLUMN created_by TEXT DEFAULT 'admin'")
        
        if 'isolation_level' not in column_names:
            logger.info("Adding isolation_level column")
            cursor.execute("ALTER TABLE users ADD COLUMN isolation_level TEXT DEFAULT 'air_gapped'")
        
        if 'is_locked' not in column_names:
            logger.info("Adding is_locked column")
            cursor.execute("ALTER TABLE users ADD COLUMN is_locked INTEGER DEFAULT 0")
        
        if 'failed_attempts' not in column_names:
            logger.info("Adding failed_attempts column")
            cursor.execute("ALTER TABLE users ADD COLUMN failed_attempts INTEGER DEFAULT 0")
        
        if 'lockout_until' not in column_names:
            logger.info("Adding lockout_until column")
            cursor.execute("ALTER TABLE users ADD COLUMN lockout_until TEXT")
        
        if 'last_login' not in column_names:
            logger.info("Adding last_login column")
            cursor.execute("ALTER TABLE users ADD COLUMN last_login TEXT")
        
        # Update existing users to have proper values
        cursor.execute("UPDATE users SET account_type = 'admin' WHERE role = 'admin'")
        cursor.execute("UPDATE users SET account_type = 'user' WHERE role != 'admin'")
        cursor.execute("UPDATE users SET created_by = 'admin' WHERE created_by IS NULL")
        cursor.execute("UPDATE users SET isolation_level = 'air_gapped' WHERE isolation_level IS NULL")
        cursor.execute("UPDATE users SET is_locked = 0 WHERE is_locked IS NULL")
        cursor.execute("UPDATE users SET failed_attempts = 0 WHERE failed_attempts IS NULL")
        
        conn.commit()
        
        # Check final schema
        cursor.execute("PRAGMA table_info(users)")
        final_columns = cursor.fetchall()
        final_column_names = [col[1] for col in final_columns]
        
        logger.info(f"Final columns: {final_column_names}")
        
        conn.close()
        
        logger.info("✅ Database schema fixed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error fixing database schema: {e}")
        return False

def list_users_after_fix():
    """List users after fixing the schema."""
    try:
        conn = sqlite3.connect("security_keys/users.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, role, account_type, created_at, last_login, 
                   failed_attempts, is_locked, lockout_until, created_by, isolation_level
            FROM users ORDER BY created_at DESC
        ''')
        
        users = cursor.fetchall()
        conn.close()
        
        logger.info("📋 Users after schema fix:")
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
            logger.info(f"👤 Created By: {user[9]}")
            logger.info(f"🔐 Isolation Level: {user[10]}")
            logger.info("-" * 40)
        
        return users
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return []

def convert_all_to_admin():
    """Convert all users to admin role."""
    try:
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

def main():
    """Main function to fix database and convert users."""
    logger.info("🔧 Starting Database Schema Fix")
    logger.info("=" * 50)
    
    # Step 1: Fix database schema
    logger.info("🔧 Step 1: Fixing database schema")
    success = fix_users_database()
    
    if not success:
        logger.error("❌ Failed to fix database schema")
        return
    
    # Step 2: List users after fix
    logger.info("\n📋 Step 2: Listing users after schema fix")
    list_users_after_fix()
    
    # Step 3: Convert all users to admin
    logger.info("\n👑 Step 3: Converting all users to admin")
    success, message = convert_all_to_admin()
    
    if success:
        logger.info(f"✅ {message}")
    else:
        logger.error(f"❌ {message}")
        return
    
    # Step 4: List final users
    logger.info("\n📋 Step 4: Listing final users")
    list_users_after_fix()
    
    logger.info("\n🎉 Database Schema Fix Complete!")
    logger.info("=" * 50)
    logger.info("✅ Database schema fixed")
    logger.info("✅ All users converted to admin role")
    logger.info("✅ All columns properly set")
    logger.info("✅ System ready for testing")

if __name__ == "__main__":
    main() 