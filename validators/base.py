"""
Base validator class providing common validation functionality.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class ValidationError:
    """Represents a validation error."""
    severity: str  # 'error', 'warning', 'info'
    field: str
    message: str
    value: Any = None
    
    def __str__(self):
        if self.value is not None:
            return f"[{self.severity.upper()}] {self.field}: {self.message} (value: {self.value})"
        return f"[{self.severity.upper()}] {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Results of a validation operation."""
    validator_name: str
    data_type: str
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info: List[ValidationError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        return len(self.warnings)
    
    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)
    
    def add_error(self, field: str, message: str, value: Any = None):
        """Add an error to the validation result."""
        self.errors.append(ValidationError('error', field, message, value))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add a warning to the validation result."""
        self.warnings.append(ValidationError('warning', field, message, value))
    
    def add_info(self, field: str, message: str, value: Any = None):
        """Add an info message to the validation result."""
        self.info.append(ValidationError('info', field, message, value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'validator_name': self.validator_name,
            'data_type': self.data_type,
            'is_valid': self.is_valid,
            'timestamp': self.timestamp.isoformat(),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'errors': [str(e) for e in self.errors],
            'warnings': [str(w) for w in self.warnings],
            'info': [str(i) for i in self.info],
            'metadata': self.metadata,
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "✅ PASSED" if self.is_valid else "❌ FAILED"
        lines = [
            f"\n{'='*60}",
            f"Validation Report: {self.validator_name}",
            f"{'='*60}",
            f"Status: {status}",
            f"Data Type: {self.data_type}",
            f"Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
        ]
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        if self.errors:
            lines.append("\n❌ ERRORS:")
            for error in self.errors:
                lines.append(f"  • {error}")
        
        if self.warnings:
            lines.append("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  • {warning}")
        
        if self.info:
            lines.append("\nℹ️  INFO:")
            for info_msg in self.info:
                lines.append(f"  • {info_msg}")
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


class BaseValidator:
    """Base class for all validators."""
    
    def __init__(self, league_id: str, year: str, division: Optional[str] = None):
        self.league_id = league_id
        self.year = year
        self.division = division
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame.
        
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate()")
    
    def _check_not_empty(self, df: pd.DataFrame, result: ValidationResult) -> bool:
        """Check that DataFrame is not empty."""
        if df is None:
            result.add_error('dataframe', 'DataFrame is None')
            return False
        
        if df.empty:
            result.add_warning('dataframe', 'DataFrame is empty (0 rows)')
            return False
        
        result.add_info('dataframe', f'DataFrame has {len(df)} rows')
        return True
    
    def _check_required_columns(self, df: pd.DataFrame, required_columns: Set[str], 
                                result: ValidationResult) -> bool:
        """Check that all required columns are present."""
        actual_columns = set(df.columns)
        missing = required_columns - actual_columns
        
        if missing:
            result.add_error('columns', f'Missing required columns: {missing}')
            return False
        
        result.add_info('columns', f'All required columns present: {required_columns}')
        return True
    
    def _check_no_null_values(self, df: pd.DataFrame, columns: Set[str], 
                              result: ValidationResult) -> bool:
        """Check that specified columns have no null values."""
        all_valid = True
        
        for col in columns:
            if col not in df.columns:
                continue
            
            null_count = df[col].isna().sum()
            if null_count > 0:
                result.add_error(col, f'Contains {null_count} null values')
                all_valid = False
        
        if all_valid:
            result.add_info('null_check', f'No null values in required columns: {columns}')
        
        return all_valid
    
    def _check_data_types(self, df: pd.DataFrame, expected_types: Dict[str, str],
                         result: ValidationResult) -> bool:
        """Check that columns have expected data types."""
        all_valid = True
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                continue
            
            actual_type = df[col].dtype
            
            # Map pandas dtypes to expected type strings
            type_mapping = {
                'int': ['int32', 'int64', 'Int32', 'Int64'],
                'float': ['float32', 'float64'],
                'str': ['object', 'string'],
                'datetime': ['datetime64[ns]', 'datetime64'],
            }
            
            if expected_type in type_mapping:
                valid_types = type_mapping[expected_type]
                if str(actual_type) not in valid_types:
                    result.add_warning(col, 
                        f'Expected type {expected_type}, got {actual_type}')
                    all_valid = False
        
        return all_valid
    
    def _check_value_range(self, df: pd.DataFrame, column: str, min_val: Any, 
                          max_val: Any, result: ValidationResult) -> bool:
        """Check that values in a column are within expected range."""
        if column not in df.columns:
            return True
        
        series = df[column].dropna()
        if len(series) == 0:
            return True
        
        # Try to convert to numeric if needed (handles string columns)
        if series.dtype == 'object':
            try:
                series = pd.to_numeric(series, errors='coerce')
                series = series.dropna()  # Drop any values that couldn't be converted
            except Exception:
                # If conversion fails, skip validation for this column
                result.add_warning(column, 
                    f'Column contains non-numeric values, skipping range validation')
                return True
        
        if len(series) == 0:
            return True
        
        all_valid = True
        
        if min_val is not None:
            below_min = (series < min_val).sum()
            if below_min > 0:
                result.add_error(column, 
                    f'{below_min} values below minimum {min_val}')
                all_valid = False
        
        if max_val is not None:
            above_max = (series > max_val).sum()
            if above_max > 0:
                result.add_error(column,
                    f'{above_max} values above maximum {max_val}')
                all_valid = False
        
        return all_valid
    
    def _check_row_count_range(self, df: pd.DataFrame, min_rows: int, max_rows: int,
                               result: ValidationResult) -> bool:
        """Check that row count is within expected range."""
        row_count = len(df)
        
        if row_count < min_rows:
            result.add_warning('row_count',
                f'Row count {row_count} below expected minimum {min_rows}')
            return False
        
        if row_count > max_rows:
            result.add_warning('row_count',
                f'Row count {row_count} above expected maximum {max_rows}')
            return False
        
        result.add_info('row_count',
            f'Row count {row_count} within expected range [{min_rows}, {max_rows}]')
        return True
    
    def _check_unique_values(self, df: pd.DataFrame, columns: List[str],
                            result: ValidationResult) -> bool:
        """Check for duplicate values in specified columns."""
        all_valid = True
        
        for col in columns:
            if col not in df.columns:
                continue
            
            duplicates = df[col].duplicated().sum()
            if duplicates > 0:
                result.add_warning(col, f'Contains {duplicates} duplicate values')
                all_valid = False
        
        return all_valid
    
    def _log_result(self, result: ValidationResult):
        """Log validation result."""
        if result.is_valid:
            self.logger.info(f"✅ Validation passed: {result.validator_name}")
        else:
            self.logger.error(f"❌ Validation failed: {result.validator_name} "
                            f"({result.error_count} errors, {result.warning_count} warnings)")
            for error in result.errors:
                self.logger.error(f"  {error}")
            for warning in result.warnings:
                self.logger.warning(f"  {warning}")
