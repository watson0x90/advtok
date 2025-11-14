"""
AdvTok Vulnerability Scanner

A comprehensive security scanner for detecting tokenization-based vulnerabilities
in Large Language Models (LLMs).

This scanner tests for:
- Adversarial tokenization bypass (AdvTok attacks)
- Chat template configuration issues
- Unicode normalization vulnerabilities
- Whitespace manipulation
- State isolation problems
- API security misconfigurations

Provides actionable, research-backed recommendations for fixing vulnerabilities.

Example usage:
    from scanner import AdvTokScanner

    scanner = AdvTokScanner("meta-llama/Llama-3.2-1B-Instruct")
    scanner.load_model()
    scanner.run_scan()
    scanner.generate_report()
"""

__version__ = "1.0.0"
__author__ = "AdvTok Research Team"

from .scanner import AdvTokScanner
from .detectors import (
    VulnerabilityFinding,
    TokenizationBypassDetector,
    ChatTemplateDetector,
    UnicodeNormalizationDetector,
    WhitespaceManipulationDetector,
    StateIsolationDetector,
    TokenInputAcceptanceDetector
)
from .recommendations import get_recommendation, get_recommendations_by_priority

__all__ = [
    'AdvTokScanner',
    'VulnerabilityFinding',
    'TokenizationBypassDetector',
    'ChatTemplateDetector',
    'UnicodeNormalizationDetector',
    'WhitespaceManipulationDetector',
    'StateIsolationDetector',
    'TokenInputAcceptanceDetector',
    'get_recommendation',
    'get_recommendations_by_priority'
]
