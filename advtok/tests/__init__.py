"""
AdvTok Test Suite
==================

Comprehensive unit and integration tests for the AdvTok package.

Test Modules:
-------------
- test_smoke.py: Quick validation tests (~1 second)
- test_advtok_stability.py: Comprehensive stability tests (~15 seconds)

Usage:
------
Run from the advtok directory:

    # Quick validation
    python tests/test_smoke.py

    # Comprehensive tests
    python tests/test_advtok_stability.py

    # Using unittest directly
    python -m unittest tests.test_smoke
    python -m unittest tests.test_advtok_stability

    # Run all tests
    python -m unittest discover tests

For more information, see tests/README.md
"""

__version__ = "1.1.0"
__all__ = ["test_smoke", "test_advtok_stability"]
