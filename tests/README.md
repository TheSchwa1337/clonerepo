# Tests Directory

This directory contains all test files for the Schwabot system.

## Structure
- `integration/`: Integration tests that test multiple components together
- `unit/`: Unit tests for individual components
- `performance/`: Performance and load tests
- `security/`: Security and vulnerability tests

## Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/integration/
python -m pytest tests/unit/
python -m pytest tests/performance/
python -m pytest tests/security/
```
