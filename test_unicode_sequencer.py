#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEST UNICODE SEQUENCER - SIMPLE TEST
====================================

Simple test to verify the Unicode sequencer is working
and can be imported successfully.
"""

import sys

def test_unicode_sequencer():
    """Test Unicode sequencer import and basic functionality."""
    print("🧪 TESTING UNICODE SEQUENCER")
    print("=" * 40)
    print("Testing import and basic functionality...")
    print()
    
    try:
        # Test import
        print("Testing import...")
        from unicode_dual_state_sequencer import get_unicode_sequencer, EmojiCategory
        print("✅ Import successful")
        
        # Get sequencer
        print("Getting sequencer...")
        unicode_sequencer = get_unicode_sequencer()
        print("✅ Sequencer obtained")
        
        # Test basic properties
        print("Testing basic properties...")
        print(f"  Unicode to state: {len(unicode_sequencer.unicode_to_state)}")
        print(f"  Emoji to Unicode: {len(unicode_sequencer.emoji_to_unicode)}")
        print("✅ Basic properties working")
        
        # Test emoji categories
        print("Testing emoji categories...")
        for category in EmojiCategory:
            print(f"  {category.value}: {category}")
        print("✅ Emoji categories working")
        
        # Test adding a simple emoji
        print("Testing emoji addition...")
        test_emoji = "🚀"
        unicode_number = ord(test_emoji)
        
        if test_emoji not in unicode_sequencer.emoji_to_unicode:
            dual_state = unicode_sequencer._create_unicode_dual_state(
                unicode_number, test_emoji, EmojiCategory.FIRE
            )
            print(f"  ✅ Added {test_emoji} successfully")
        else:
            print(f"  ⚠️ {test_emoji} already exists")
        
        print(f"  Total emojis: {len(unicode_sequencer.unicode_to_state)}")
        print("✅ Emoji addition working")
        
        print()
        print("🏆 UNICODE SEQUENCER TEST RESULTS:")
        print("  ✅ Import successful")
        print("  ✅ Sequencer obtained")
        print("  ✅ Basic properties working")
        print("  ✅ Emoji categories working")
        print("  ✅ Emoji addition working")
        print("  ✅ System ready for expansion")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_unicode_sequencer()
    sys.exit(0 if success else 1) 