"""
Data validation package for SquashApp.

Provides validators for all scraped data types to ensure data quality
and catch issues early.
"""

from .base import BaseValidator, ValidationError, ValidationResult
from .players import PlayersValidator
from .ranking import RankingValidator
from .reports import ValidationReport, validate_all_division_data
from .schedules import SchedulesValidator
from .summary import SummaryValidator
from .teams import TeamsValidator

__all__ = [
    "BaseValidator",
    "ValidationResult",
    "ValidationError",
    "TeamsValidator",
    "SummaryValidator",
    "SchedulesValidator",
    "RankingValidator",
    "PlayersValidator",
    "ValidationReport",
    "validate_all_division_data",
]
