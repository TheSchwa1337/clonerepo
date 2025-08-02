# 🪙 COINBASE API SETUP GUIDE - SCHWABOT
==========================================

## 🎯 **Complete Coinbase Integration for Schwabot**

**UPDATE**: Coinbase Pro has been fully integrated into Coinbase! 🎉
- ✅ **Unified Coinbase Exchange** - All trading now through single platform
- ✅ **Current CCXT Integration** - Using `ccxt.coinbase` (not deprecated `coinbasepro`)
- ✅ **Real-time API Access** - Live pricing and trading capabilities
- ✅ **Advanced Features** - All Pro features now available in main Coinbase

This guide ensures **100% current Coinbase compatibility** across ALL Schwabot systems:
- ✅ API Key Configuration System
- ✅ Real API Pricing System  
- ✅ Live Market Data Integration
- ✅ CCXT Integration (Current)
- ✅ Mathematical Pipeline Integration
- ✅ Timing Drift Protocol
- ✅ Phantom Relation Ghost Protocol
- ✅ Memory Storage Systems
- ✅ USB Security Integration
- ✅ 5-Layer Encryption System

---

## 🔑 **Step 1: Get Coinbase API Keys**

### 1.1 Login to Coinbase
- Go to: https://www.coinbase.com/settings/api
- Login with your Coinbase account (unified platform)

### 1.2 Create API Key
- Click **"New API Key"**
- Set permissions:
  - ✅ **View** (required)
  - ✅ **Trade** (for live trading)
  - ✅ **Transfer** (optional)

### 1.3 Save Your Credentials
You'll receive:
- **API Key** (starts with `12345678-1234-1234-1234-123456789abc`)
- **Secret Key** (long string of letters/numbers)
- **Passphrase** (you create this - remember it!)

⚠️ **IMPORTANT**: Save these securely - you won't see them again!

---

## 🔧 **Step 2: Configure Schwabot**

### 2.1 Run API Configuration
```bash
python api_key_configuration.py
```

### 2.2 Select Option 1: "Configure Trading Exchange Keys"

### 2.3 Enter Coinbase Credentials
When prompted for Coinbase:
- **API Key**: Paste your Coinbase API key
- **Secret Key**: Paste your Coinbase secret key  
- **Passphrase**: Enter your passphrase

### 2.4 Verify Configuration
The system will:
- ✅ Encrypt your keys with 5-layer encryption
- ✅ Store them securely in USB memory
- ✅ Test the connection with current unified Coinbase

---

## 🧪 **Step 3: Test Integration**

### 3.1 Run Comprehensive Test
```bash
python test_coinbase_integration.py
```

### 3.2 Expected Results
All tests should show:
- ✅ API Key Configuration: PASSED
- ✅ Real API Pricing: PASSED
- ✅ Live Market Data: PASSED
- ✅ CCXT Integration: PASSED (Current unified exchange)
- ✅ Mathematical Pipeline: PASSED
- ✅ Timing Drift Protocol: PASSED
- ✅ Phantom Relation Ghost Protocol: PASSED
- ✅ Memory Storage: PASSED
- ✅ USB Security: PASSED
- ✅ 5-Layer Encryption: PASSED

---

## 🔐 **Security Features**

### 5-Layer Encryption System
Your Coinbase keys are protected by:
1. **Alpha256 Encryption** (Advanced 256-bit)
2. **Alpha Encryption (Ω-B-Γ Logic)** (Mathematical security)
3. **AES-256 Encryption** (Production grade)
4. **Alpha Encryption Fixed** (Additional layer)
5. **Base64 Encoding** (Final layer)

### USB Security
- Keys automatically backed up to USB drive
- Encrypted storage on portable device
- Automatic synchronization

---

## 📊 **Coinbase Integration Status**

### ✅ **Fully Integrated Systems:**

#### 🔑 **API Key Configuration**
- Current unified Coinbase API key management
- Passphrase support
- 5-layer encryption
- USB backup/restore

#### 📈 **Real API Pricing**
- Live unified Coinbase price feeds
- Real-time market data
- Automatic fallback systems
- Memory caching

#### 🔌 **CCXT Integration (Current)**
- **Current unified Coinbase exchange** (`ccxt.coinbase`)
- **No deprecated coinbasepro** - fully updated
- Rate limiting
- Error handling

#### 🧮 **Mathematical Pipeline**
- Real unified Coinbase prices in calculations
- Decimal key extraction
- Price ratio analysis
- Strategy integration

#### ⏰ **Timing Drift Protocol**
- Unified Coinbase API timing synchronization
- Latency monitoring
- Drift detection
- Performance optimization

#### 👻 **Phantom Relation Ghost Protocol**
- Cross-exchange price comparison
- Unified Coinbase vs other exchanges
- Anomaly detection
- Relationship analysis

#### 💾 **Memory Storage**
- Unified Coinbase data persistence
- Historical price storage
- Performance tracking
- USB synchronization

#### 🔒 **USB Security**
- Encrypted unified Coinbase data on USB
- Automatic backup
- Portable security
- Disaster recovery

---

## 🚀 **Usage Examples**

### Get Real Unified Coinbase Prices
```python
from real_api_pricing_memory_system import get_real_price_data

# Get BTC price from unified Coinbase
btc_price = get_real_price_data('BTC/USD', 'coinbase')
print(f"BTC Price: ${btc_price:,.2f}")

# Get ETH price from unified Coinbase
eth_price = get_real_price_data('ETH/USD', 'coinbase')
print(f"ETH Price: ${eth_price:,.2f}")
```

### Store Unified Coinbase Data
```python
from real_api_pricing_memory_system import store_memory_entry

# Store unified Coinbase trading data
store_memory_entry(
    data_type='coinbase_trade',
    data={
        'symbol': 'BTC/USD',
        'price': 50000.0,
        'action': 'buy',
        'timestamp': '2025-01-17T20:30:00Z'
    },
    source='coinbase_trading',
    priority=2,
    tags=['coinbase', 'trading', 'btc', 'unified']
)
```

---

## 🔧 **Troubleshooting**

### Issue: "401 Unauthorized"
**Solution**: Check your API credentials
1. Verify API key is correct
2. Verify secret key is correct  
3. Verify passphrase is correct
4. Check API permissions

### Issue: "Rate limit exceeded"
**Solution**: Wait and retry
- Unified Coinbase has rate limits
- System automatically handles this
- Wait 1-2 minutes between requests

### Issue: "Passphrase not found"
**Solution**: Reconfigure API keys
```bash
python api_key_configuration.py
```
- Select option 1
- Re-enter all unified Coinbase credentials including passphrase

### Issue: "coinbasepro not found"
**Solution**: Update CCXT library
```bash
pip install ccxt --upgrade
```
- This ensures you have the current unified Coinbase exchange

---

## 📞 **Support**

If you encounter issues:
1. Run the comprehensive test: `python test_coinbase_integration.py`
2. Check the logs: `coinbase_integration_test.log`
3. Verify your unified Coinbase API credentials
4. Ensure you have the latest CCXT library: `pip install ccxt --upgrade`

---

## 🎉 **Success Indicators**

When unified Coinbase is fully integrated, you'll see:
- ✅ All 10 integration tests pass
- ✅ Real-time unified Coinbase prices in all systems
- ✅ Secure 5-layer encryption of your keys
- ✅ USB backup of all unified Coinbase data
- ✅ Perfect integration with mathematical pipelines
- ✅ No timing drift or phantom relationships
- ✅ **Current CCXT integration** (no deprecated coinbasepro)

**🎯 Unified Coinbase is now your primary exchange with bulletproof current integration!** 