"""
Shellbert Evaluation System

Pre-deployment testing and evaluation of:
- Personality consistency and expression
- Safety and ethical behavior
- Tool usage capabilities  
- Memory functionality
- EA knowledge and reasoning
- General capabilities

Uses the `inspect` library for comprehensive evaluation frameworks.
"""

from .evaluation_runner import EvaluationRunner, EvaluationSuite

__all__ = [
    'EvaluationRunner',
    'EvaluationSuite'
]