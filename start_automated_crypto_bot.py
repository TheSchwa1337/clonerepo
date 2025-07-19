#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AUTOMATED CRYPTO TRADING BOT LAUNCHER
========================================

Simple launcher for the Schwabot Automated Crypto Trading Bot.
This script provides easy configuration and startup for the complete automated trading system.

Usage:
    python start_automated_crypto_bot.py --demo                    # Start in demo mode
    python start_automated_crypto_bot.py --live --config config.yaml  # Start live trading
    python start_automated_crypto_bot.py --backtest --days 30      # Run backtest
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the automated bot
from core.automated_crypto_trading_bot import (
    AutomatedCryptoTradingBot,
    AutomatedBotConfig,
    TradingMode,
    create_automated_crypto_trading_bot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('automated_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return {}

def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the bot."""
    return {
        'trading_mode': 'demo',
        'initial_capital': 10000.0,
        'max_position_size': 0.1,
        'min_position_size': 0.01,
        'stop_loss_percentage': 0.02,
        'take_profit_percentage': 0.05,
        'max_daily_loss': 0.05,
        'max_drawdown': 0.15,
        'rebalancing_enabled': True,
        'rebalancing_threshold': 0.05,
        'rebalancing_interval': 3600,
        'target_allocation': {
            'BTC': 0.4,
            'ETH': 0.3,
            'USDC': 0.3
        },
        'trading_pairs': [
            'BTC/USDC',
            'ETH/USDC',
            'SOL/USDC'
        ],
        'exchanges': {
            'binance': {
                'enabled': True,
                'sandbox': True,
                'api_key': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET', '')
            },
            'coinbase': {
                'enabled': True,
                'sandbox': True,
                'api_key': os.getenv('COINBASE_API_KEY', ''),
                'secret': os.getenv('COINBASE_SECRET', '')
            }
        },
        'math_confidence_threshold': 0.7,
        'math_risk_threshold': 0.8,
        'execution_timeout': 30,
        'max_retry_attempts': 3,
        'slippage_tolerance': 0.001,
        'performance_tracking': True,
        'enable_alerts': True,
        'log_level': 'INFO'
    }

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        logger.info(f"✅ Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"❌ Failed to save configuration: {e}")

def create_bot_config(args) -> AutomatedBotConfig:
    """Create bot configuration from command line arguments and config file."""
    # Load config file if provided
    config_data = {}
    if args.config:
        config_data = load_config(args.config)
    
    # Use default config if no file provided
    if not config_data:
        config_data = create_default_config()
    
    # Override with command line arguments
    if args.demo:
        config_data['trading_mode'] = 'demo'
    elif args.live:
        config_data['trading_mode'] = 'live'
    elif args.backtest:
        config_data['trading_mode'] = 'backtest'
    
    if args.capital:
        config_data['initial_capital'] = args.capital
    
    if args.pairs:
        config_data['trading_pairs'] = args.pairs.split(',')
    
    # Create bot config
    trading_mode = TradingMode(config_data.get('trading_mode', 'demo'))
    
    return AutomatedBotConfig(
        trading_mode=trading_mode,
        initial_capital=config_data.get('initial_capital', 10000.0),
        max_position_size=config_data.get('max_position_size', 0.1),
        min_position_size=config_data.get('min_position_size', 0.01),
        stop_loss_percentage=config_data.get('stop_loss_percentage', 0.02),
        take_profit_percentage=config_data.get('take_profit_percentage', 0.05),
        max_daily_loss=config_data.get('max_daily_loss', 0.05),
        max_drawdown=config_data.get('max_drawdown', 0.15),
        rebalancing_enabled=config_data.get('rebalancing_enabled', True),
        rebalancing_threshold=config_data.get('rebalancing_threshold', 0.05),
        rebalancing_interval=config_data.get('rebalancing_interval', 3600),
        target_allocation=config_data.get('target_allocation', {
            'BTC': 0.4,
            'ETH': 0.3,
            'USDC': 0.3
        }),
        trading_pairs=config_data.get('trading_pairs', [
            'BTC/USDC',
            'ETH/USDC',
            'SOL/USDC'
        ]),
        exchanges=config_data.get('exchanges', {}),
        math_confidence_threshold=config_data.get('math_confidence_threshold', 0.7),
        math_risk_threshold=config_data.get('math_risk_threshold', 0.8),
        execution_timeout=config_data.get('execution_timeout', 30),
        max_retry_attempts=config_data.get('max_retry_attempts', 3),
        slippage_tolerance=config_data.get('slippage_tolerance', 0.001),
        performance_tracking=config_data.get('performance_tracking', True),
        enable_alerts=config_data.get('enable_alerts', True),
        log_level=config_data.get('log_level', 'INFO')
    )

async def run_bot(bot: AutomatedCryptoTradingBot, duration: int = None):
    """Run the bot for a specified duration or indefinitely."""
    try:
        # Start the bot
        await bot.start()
        
        logger.info("🤖 Automated Crypto Trading Bot is running...")
        logger.info("Press Ctrl+C to stop the bot")
        
        # Run for specified duration or indefinitely
        if duration:
            logger.info(f"Bot will run for {duration} seconds")
            await asyncio.sleep(duration)
            await bot.stop()
        else:
            # Run indefinitely
            while True:
                await asyncio.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("🛑 Received stop signal")
        await bot.stop()
    except Exception as e:
        logger.error(f"❌ Error running bot: {e}")
        await bot.stop()
        raise

async def run_backtest(bot: AutomatedCryptoTradingBot, days: int):
    """Run backtest for specified number of days."""
    try:
        logger.info(f"📊 Starting backtest for {days} days")
        
        # Start the bot
        await bot.start()
        
        # Run for the specified duration
        duration_seconds = days * 24 * 3600  # Convert days to seconds
        await asyncio.sleep(duration_seconds)
        
        # Stop the bot
        await bot.stop()
        
        # Get final results
        status = bot.get_bot_status()
        trade_history = bot.get_trade_history()
        rebalancing_history = bot.get_rebalancing_history()
        
        # Print results
        print("\n" + "="*50)
        print("📊 BACKTEST RESULTS")
        print("="*50)
        print(f"Duration: {days} days")
        print(f"Total Trades: {status['performance']['total_trades']}")
        print(f"Win Rate: {status['performance']['win_rate']:.2%}")
        print(f"Total P&L: ${status['performance']['total_pnl']:.2f} ({status['performance']['total_pnl_percentage']:.2%})")
        print(f"Max Drawdown: {status['performance']['max_drawdown']:.2%}")
        print(f"Rebalancing Count: {status['performance']['rebalancing_count']}")
        print(f"Final Portfolio Value: ${status.get('final_value', 0):.2f}")
        print("="*50)
        
        # Save results to file
        results = {
            'backtest_duration_days': days,
            'status': status,
            'trade_history': trade_history,
            'rebalancing_history': rebalancing_history,
            'timestamp': bot.performance.start_time.isoformat()
        }
        
        with open(f'backtest_results_{days}days.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Backtest results saved to backtest_results_{days}days.json")
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        await bot.stop()
        raise

def setup_alert_callbacks(bot: AutomatedCryptoTradingBot):
    """Setup alert callbacks for the bot."""
    
    async def alert_callback(alert: Dict[str, Any]):
        """Handle alert events."""
        alert_type = alert.get('type', 'unknown')
        data = alert.get('data', {})
        
        if alert_type == 'significant_pnl_change':
            pnl_pct = data.get('pnl_percentage', 0)
            direction = "📈 PROFIT" if pnl_pct > 0 else "📉 LOSS"
            print(f"\n🚨 {direction} ALERT: {pnl_pct:.2%} P&L change")
            
        elif alert_type == 'high_drawdown':
            drawdown = data.get('max_drawdown', 0)
            print(f"\n🚨 HIGH DRAWDOWN ALERT: {drawdown:.2%} drawdown")
            
        elif alert_type == 'low_win_rate':
            win_rate = data.get('win_rate', 0)
            print(f"\n🚨 LOW WIN RATE ALERT: {win_rate:.2%} win rate")
            
        elif alert_type == 'emergency_stop':
            print(f"\n🚨 EMERGENCY STOP TRIGGERED: {data}")
    
    bot.add_alert_callback(alert_callback)

def setup_trade_callbacks(bot: AutomatedCryptoTradingBot):
    """Setup trade callbacks for the bot."""
    
    async def trade_callback(event_type: str, data: Any, metadata: Dict[str, Any] = None):
        """Handle trade events."""
        if event_type == 'trade_executed':
            symbol = data.get('symbol', 'Unknown')
            side = data.get('side', 'Unknown')
            amount = data.get('amount', 0)
            price = data.get('price', 0)
            pnl = data.get('pnl', 0)
            
            emoji = "💰" if pnl > 0 else "💸"
            print(f"\n{emoji} TRADE EXECUTED: {side} {amount} {symbol} @ ${price:.2f} (P&L: ${pnl:.2f})")
    
    bot.add_trade_callback(trade_callback)

def main():
    """Main function to parse arguments and run the bot."""
    parser = argparse.ArgumentParser(
        description="Schwabot Automated Crypto Trading Bot Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_automated_crypto_bot.py --demo                    # Start in demo mode
  python start_automated_crypto_bot.py --live --config config.yaml  # Start live trading
  python start_automated_crypto_bot.py --backtest --days 30      # Run 30-day backtest
  python start_automated_crypto_bot.py --demo --capital 5000     # Demo with $5000 capital
  python start_automated_crypto_bot.py --create-config           # Create default config file
        """
    )
    
    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--demo', action='store_true', help='Run in demo mode (paper trading)')
    mode_group.add_argument('--live', action='store_true', help='Run in live trading mode')
    mode_group.add_argument('--backtest', action='store_true', help='Run backtest')
    mode_group.add_argument('--create-config', action='store_true', help='Create default configuration file')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--capital', type=float, help='Initial capital amount')
    parser.add_argument('--pairs', type=str, help='Comma-separated trading pairs (e.g., BTC/USDC,ETH/USDC)')
    
    # Backtest arguments
    parser.add_argument('--days', type=int, default=30, help='Number of days for backtest (default: 30)')
    
    # Runtime arguments
    parser.add_argument('--duration', type=int, help='Run bot for specified number of seconds')
    
    args = parser.parse_args()
    
    # Handle create-config mode
    if args.create_config:
        config = create_default_config()
        config_path = 'automated_bot_config.yaml'
        save_config(config, config_path)
        print(f"✅ Default configuration created: {config_path}")
        print("Edit the file to customize settings before running the bot.")
        return
    
    try:
        # Create bot configuration
        bot_config = create_bot_config(args)
        
        # Create bot
        bot = create_automated_crypto_trading_bot(bot_config)
        
        # Setup callbacks
        setup_alert_callbacks(bot)
        setup_trade_callbacks(bot)
        
        # Print configuration summary
        print("\n" + "="*50)
        print("🤖 SCHWABOT AUTOMATED CRYPTO TRADING BOT")
        print("="*50)
        print(f"Mode: {bot_config.trading_mode.value.upper()}")
        print(f"Initial Capital: ${bot_config.initial_capital:,.2f}")
        print(f"Trading Pairs: {', '.join(bot_config.trading_pairs)}")
        print(f"Target Allocation: {bot_config.target_allocation}")
        print(f"Stop Loss: {bot_config.stop_loss_percentage:.1%}")
        print(f"Take Profit: {bot_config.take_profit_percentage:.1%}")
        print(f"Max Daily Loss: {bot_config.max_daily_loss:.1%}")
        print(f"Rebalancing: {'Enabled' if bot_config.rebalancing_enabled else 'Disabled'}")
        print("="*50)
        
        # Run the bot
        if args.backtest:
            asyncio.run(run_backtest(bot, args.days))
        else:
            asyncio.run(run_bot(bot, args.duration))
        
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 