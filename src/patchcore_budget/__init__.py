"""PatchCore-style memory allocation research toolkit."""

from .model import BudgetMemory
from .selection import ExperimentConfig, select_models

__all__ = ["BudgetMemory", "ExperimentConfig", "select_models"]
__version__ = "0.1.0"
