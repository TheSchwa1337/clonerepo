# 🤝 Contributing to Schwabot

Thank you for your interest in contributing to Schwabot! This guide will help you get started with contributing to the project.

## 🎯 How to Contribute

We welcome contributions from the community in many forms:

- **🐛 Bug Reports**: Report bugs and issues
- **💡 Feature Requests**: Suggest new features and improvements
- **📝 Documentation**: Improve or add documentation
- **🔧 Code Contributions**: Submit code improvements and new features
- **🧪 Testing**: Help test the system and report issues
- **🌐 Localization**: Help translate the interface
- **📊 Performance**: Optimize performance and efficiency

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of Python and trading systems
- Familiarity with the Schwabot architecture

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/schwabot.git
   cd schwabot
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Configure Development Settings**
   ```bash
   # Copy development config
   cp AOI_Base_Files_Schwabot/config/config.example.yaml AOI_Base_Files_Schwabot/config/config.dev.yaml
   
   # Edit development config
   nano AOI_Base_Files_Schwabot/config/config.dev.yaml
   ```

4. **Run Tests**
   ```bash
   # Run all tests
   python run_tests.py
   
   # Run specific test categories
   python -m pytest tests/unit/
   python -m pytest tests/integration/
   ```

## 📋 Development Workflow

### 1. Issue Tracking

Before starting work:

1. **Check Existing Issues**: Search for existing issues related to your work
2. **Create New Issue**: If no existing issue, create a new one describing your contribution
3. **Discuss Approach**: Engage with maintainers to discuss the best approach
4. **Get Assignment**: Get assigned to the issue if it's a bug fix or feature

### 2. Branch Strategy

```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/your-bug-description
```

### 3. Development Process

1. **Make Changes**: Implement your changes following coding standards
2. **Test Locally**: Run tests and verify functionality
3. **Update Documentation**: Update relevant documentation
4. **Commit Changes**: Use conventional commit messages

### 4. Testing

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/performance/ -v
python -m pytest tests/security/ -v

# Run with coverage
python -m pytest --cov=AOI_Base_Files_Schwabot tests/

# Run linting
flake8 AOI_Base_Files_Schwabot/
black --check AOI_Base_Files_Schwabot/
```

### 5. Code Quality

```bash
# Format code
black AOI_Base_Files_Schwabot/

# Sort imports
isort AOI_Base_Files_Schwabot/

# Check for security issues
bandit -r AOI_Base_Files_Schwabot/

# Type checking
mypy AOI_Base_Files_Schwabot/
```

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear, descriptive variable names
def calculate_portfolio_value(positions, current_prices):
    """Calculate total portfolio value."""
    total_value = 0.0
    for symbol, quantity in positions.items():
        if symbol in current_prices:
            total_value += quantity * current_prices[symbol]
    return total_value

# Bad: Unclear names and no docstring
def calc_val(pos, prices):
    val = 0
    for s, q in pos.items():
        if s in prices:
            val += q * prices[s]
    return val
```

### File Organization

```
AOI_Base_Files_Schwabot/
├── core/                    # Core system components
│   ├── __init__.py
│   ├── trading_engine.py
│   ├── risk_manager.py
│   └── ...
├── gui/                     # User interface
│   ├── __init__.py
│   ├── web_interface.py
│   └── ...
├── api/                     # API integrations
│   ├── __init__.py
│   ├── exchange_api.py
│   └── ...
└── config/                  # Configuration files
    ├── config.yaml
    └── ...
```

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, request

# Local imports
from .core.trading_engine import TradingEngine
from .utils.helpers import format_currency
```

### Documentation Standards

#### Function Documentation

```python
def calculate_risk_metrics(portfolio: Dict[str, float], 
                          market_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics for a portfolio.
    
    Args:
        portfolio: Dictionary mapping symbols to position values
        market_data: DataFrame with historical market data
        
    Returns:
        Dictionary containing risk metrics:
        - volatility: Portfolio volatility
        - var_95: 95% Value at Risk
        - sharpe_ratio: Sharpe ratio
        - max_drawdown: Maximum drawdown
        
    Raises:
        ValueError: If portfolio is empty or market_data is invalid
        
    Example:
        >>> portfolio = {'BTC': 10000, 'ETH': 5000}
        >>> metrics = calculate_risk_metrics(portfolio, market_data)
        >>> print(metrics['volatility'])
        0.15
    """
    # Implementation here
    pass
```

#### Class Documentation

```python
class RiskManager:
    """
    Comprehensive risk management system for trading operations.
    
    The RiskManager provides position sizing, portfolio limits,
    and circuit breaker functionality to ensure safe trading
    operations.
    
    Attributes:
        max_position_size: Maximum position size as percentage of portfolio
        max_portfolio_risk: Maximum portfolio risk exposure
        circuit_breaker_threshold: Loss threshold for circuit breaker
        
    Example:
        >>> risk_manager = RiskManager(max_position_size=0.1)
        >>> risk_manager.validate_trade(symbol='BTC', size=1000)
        True
    """
    
    def __init__(self, max_position_size: float = 0.1):
        """
        Initialize the risk manager.
        
        Args:
            max_position_size: Maximum position size as decimal (0.1 = 10%)
        """
        self.max_position_size = max_position_size
```

## 🧪 Testing Guidelines

### Test Structure

```python
# tests/unit/test_risk_manager.py
import pytest
import numpy as np
from AOI_Base_Files_Schwabot.core.risk_manager import RiskManager

class TestRiskManager:
    """Test suite for RiskManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.risk_manager = RiskManager(max_position_size=0.1)
        self.portfolio = {'BTC': 10000, 'ETH': 5000}
    
    def test_position_size_validation(self):
        """Test position size validation."""
        # Test valid position
        assert self.risk_manager.validate_position_size('BTC', 500) == True
        
        # Test invalid position (too large)
        assert self.risk_manager.validate_position_size('BTC', 2000) == False
    
    def test_portfolio_risk_calculation(self):
        """Test portfolio risk calculation."""
        risk = self.risk_manager.calculate_portfolio_risk(self.portfolio)
        assert isinstance(risk, float)
        assert 0 <= risk <= 1
    
    @pytest.mark.parametrize("portfolio,expected", [
        ({'BTC': 1000}, 0.1),
        ({'BTC': 1000, 'ETH': 500}, 0.15),
        ({}, 0.0),
    ])
    def test_risk_calculation_variations(self, portfolio, expected):
        """Test risk calculation with various portfolios."""
        risk = self.risk_manager.calculate_portfolio_risk(portfolio)
        assert abs(risk - expected) < 0.01
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test system performance
4. **Security Tests**: Test security vulnerabilities
5. **End-to-End Tests**: Test complete workflows

### Test Coverage

- **Minimum Coverage**: 80% for new code
- **Critical Paths**: 100% coverage for risk management and trading logic
- **Documentation**: All public APIs should have tests

## 🔒 Security Guidelines

### Code Security

1. **Input Validation**: Validate all inputs
2. **Output Encoding**: Encode all outputs
3. **Secure Configuration**: Use environment variables for secrets
4. **Error Handling**: Don't expose sensitive information in errors

### Security Testing

```bash
# Run security tests
python -m pytest tests/security/

# Check for vulnerabilities
bandit -r AOI_Base_Files_Schwabot/

# Check dependencies
safety check
```

## 📊 Performance Guidelines

### Performance Testing

```bash
# Run performance tests
python -m pytest tests/performance/

# Profile code
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

### Performance Best Practices

1. **Use Efficient Data Structures**: Choose appropriate data structures
2. **Optimize Algorithms**: Use efficient algorithms
3. **Cache Results**: Cache expensive computations
4. **Profile Code**: Identify bottlenecks

## 📝 Commit Guidelines

### Conventional Commits

We use [Conventional Commits](https://www.conventionalcommits.org/):

```bash
# Feature
git commit -m "feat: add new risk management algorithm"

# Bug fix
git commit -m "fix: resolve memory leak in tensor operations"

# Documentation
git commit -m "docs: update API documentation"

# Performance
git commit -m "perf: optimize portfolio calculation"

# Breaking change
git commit -m "feat!: change API interface for risk calculation"
```

### Commit Message Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🔄 Pull Request Process

### 1. Prepare Your PR

```bash
# Ensure your branch is up to date
git checkout main
git pull origin main
git checkout your-feature-branch
git rebase main

# Run all checks
python run_tests.py
flake8 AOI_Base_Files_Schwabot/
black --check AOI_Base_Files_Schwabot/
```

### 2. Create Pull Request

1. **Title**: Clear, descriptive title
2. **Description**: Detailed description of changes
3. **Issue Link**: Link to related issue
4. **Testing**: Describe how you tested
5. **Screenshots**: If UI changes

### 3. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Security tests pass

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
```

### 4. Review Process

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code
3. **Discussion**: Address feedback and questions
4. **Approval**: Get approval from maintainers
5. **Merge**: Maintainers merge the PR

## 🏷️ Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

### Release Steps

1. **Update Version**: Update version in code
2. **Update Changelog**: Document changes
3. **Create Release**: Create GitHub release
4. **Deploy**: Deploy to production
5. **Announce**: Announce to community

## 🆘 Getting Help

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: For security issues (security@schwabot.com)

### Resources

- **Documentation**: [docs/](docs/) directory
- **Architecture**: [docs/development/architecture.md](docs/development/architecture.md)
- **API Reference**: [docs/api/](docs/api/) directory
- **Examples**: [examples/](examples/) directory

## 🎉 Recognition

### Contributors

We recognize contributors in several ways:

1. **Contributors List**: GitHub contributors page
2. **Release Notes**: Credit in release notes
3. **Documentation**: Credit in documentation
4. **Community**: Recognition in community discussions

### Contribution Levels

- **Bronze**: 1-5 contributions
- **Silver**: 6-20 contributions
- **Gold**: 21+ contributions
- **Platinum**: Major contributions and leadership

## 📄 License

By contributing to Schwabot, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Schwabot! Your contributions help make the system better for everyone.

*Last updated: July 18, 2025* 