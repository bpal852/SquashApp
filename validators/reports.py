"""
Validation report generation and management.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd

from .base import ValidationResult


class ValidationReport:
    """Manages validation reports for a scraping session."""
    
    def __init__(self, output_dir: str, year: str):
        self.output_dir = Path(output_dir)
        self.year = year
        self.results: List[ValidationResult] = []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create validation reports directory
        self.report_dir = self.output_dir / 'validation_reports'
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)
    
    def save_individual_report(self, result: ValidationResult, division: str = None):
        """Save an individual validation report."""
        # Determine filename
        if division:
            filename = f"{division}_{result.data_type}_validation.json"
        else:
            filename = f"{result.data_type}_validation.json"
        
        filepath = self.report_dir / filename
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved validation report: {filepath}")
        
        # Also save human-readable summary
        summary_file = filepath.with_suffix('.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(result.summary())
        
        return filepath
    
    def save_summary_report(self):
        """Save a summary report of all validations."""
        if not self.results:
            self.logger.warning("No validation results to report")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.report_dir / f'validation_summary_{timestamp}.txt'
        json_file = self.report_dir / f'validation_summary_{timestamp}.json'
        
        # Generate summary statistics
        total_validations = len(self.results)
        passed = sum(1 for r in self.results if r.is_valid)
        failed = total_validations - passed
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)
        
        # Write text summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"VALIDATION SUMMARY REPORT - {self.year}\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nTotal Validations: {total_validations}\n")
            f.write(f"Passed: {passed} ✅\n")
            f.write(f"Failed: {failed} ❌\n")
            f.write(f"Total Errors: {total_errors}\n")
            f.write(f"Total Warnings: {total_warnings}\n")
            f.write("\n" + "="*70 + "\n\n")
            
            # List all results
            for result in self.results:
                status_icon = "✅" if result.is_valid else "❌"
                f.write(f"{status_icon} {result.validator_name} - {result.data_type}\n")
                if result.metadata.get('division'):
                    f.write(f"   Division: {result.metadata['division']}\n")
                f.write(f"   Errors: {result.error_count}, Warnings: {result.warning_count}\n")
                
                if result.errors:
                    f.write(f"   ❌ Errors:\n")
                    for error in result.errors[:5]:  # Show first 5
                        f.write(f"      • {error}\n")
                    if result.error_count > 5:
                        f.write(f"      ... and {result.error_count - 5} more\n")
                
                if result.warnings:
                    f.write(f"   ⚠️  Warnings:\n")
                    for warning in result.warnings[:3]:  # Show first 3
                        f.write(f"      • {warning}\n")
                    if result.warning_count > 3:
                        f.write(f"      ... and {result.warning_count - 3} more\n")
                
                f.write("\n")
        
        # Write JSON summary
        summary_data = {
            'year': self.year,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_validations': total_validations,
                'passed': passed,
                'failed': failed,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        self.logger.info(f"Saved validation summary: {summary_file}")
        self.logger.info(f"Saved JSON summary: {json_file}")
        
        return summary_file
    
    def print_summary(self):
        """Print a summary to console."""
        if not self.results:
            print("No validation results")
            return
        
        passed = sum(1 for r in self.results if r.is_valid)
        failed = len(self.results) - passed
        total_errors = sum(r.error_count for r in self.results)
        total_warnings = sum(r.warning_count for r in self.results)
        
        print("\n" + "="*70)
        print(f"VALIDATION SUMMARY - {self.year}")
        print("="*70)
        print(f"Total: {len(self.results)} | Passed: {passed} ✅ | Failed: {failed} ❌")
        print(f"Errors: {total_errors} | Warnings: {total_warnings}")
        print("="*70 + "\n")
        
        if failed > 0:
            print("❌ Failed Validations:")
            for result in self.results:
                if not result.is_valid:
                    division = result.metadata.get('division', 'N/A')
                    print(f"  • {result.data_type} ({division}): "
                          f"{result.error_count} errors, {result.warning_count} warnings")
            print()
    
    def get_failed_validations(self) -> List[ValidationResult]:
        """Get all failed validation results."""
        return [r for r in self.results if not r.is_valid]
    
    def has_errors(self) -> bool:
        """Check if any validations have errors."""
        return any(r.error_count > 0 for r in self.results)
    
    def create_error_summary_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame summarizing all validation errors."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            row = {
                'validator': result.validator_name,
                'data_type': result.data_type,
                'division': result.metadata.get('division', 'N/A'),
                'status': 'PASSED' if result.is_valid else 'FAILED',
                'errors': result.error_count,
                'warnings': result.warning_count,
                'row_count': result.metadata.get('row_count', 0),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.sort_values(['status', 'errors', 'warnings'], ascending=[True, False, False])


def validate_all_division_data(output_dir: str, year: str, division: str,
                                teams_df: pd.DataFrame = None,
                                summary_df: pd.DataFrame = None,
                                schedules_df: pd.DataFrame = None,
                                ranking_df: pd.DataFrame = None,
                                players_df: pd.DataFrame = None) -> ValidationReport:
    """
    Validate all data for a division.
    
    Args:
        output_dir: Directory for validation reports
        year: Season year (e.g., "2025-2026")
        division: Division name (e.g., "M2")
        teams_df: Teams DataFrame
        summary_df: Summary DataFrame
        schedules_df: Schedules DataFrame
        ranking_df: Ranking DataFrame
        players_df: Players DataFrame
    
    Returns:
        ValidationReport with all results
    """
    from . import (TeamsValidator, SummaryValidator, SchedulesValidator,
                   RankingValidator, PlayersValidator)
    
    report = ValidationReport(output_dir, year)
    
    # Validate each DataFrame if provided
    if teams_df is not None:
        validator = TeamsValidator(league_id="", year=year, division=division)
        result = validator.validate(teams_df)
        report.add_result(result)
        report.save_individual_report(result, division)
    
    if summary_df is not None:
        validator = SummaryValidator(league_id="", year=year, division=division)
        result = validator.validate(summary_df)
        report.add_result(result)
        report.save_individual_report(result, division)
    
    if schedules_df is not None:
        validator = SchedulesValidator(league_id="", year=year, division=division)
        result = validator.validate(schedules_df)
        report.add_result(result)
        report.save_individual_report(result, division)
    
    if ranking_df is not None:
        validator = RankingValidator(league_id="", year=year, division=division)
        result = validator.validate(ranking_df)
        report.add_result(result)
        report.save_individual_report(result, division)
    
    if players_df is not None:
        validator = PlayersValidator(league_id="", year=year, division=division)
        result = validator.validate(players_df)
        report.add_result(result)
        report.save_individual_report(result, division)
    
    return report
