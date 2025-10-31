"""
Validator for team data.
"""

import pandas as pd

from .base import BaseValidator, ValidationResult


class TeamsValidator(BaseValidator):
    """Validates team roster data."""

    REQUIRED_COLUMNS = {"Team Name", "Home", "Convenor", "Email"}
    EXPECTED_MIN_TEAMS = 4  # Most divisions have at least 4 teams
    EXPECTED_MAX_TEAMS = 20  # Most divisions have at most 20 teams

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate teams DataFrame."""
        result = ValidationResult(validator_name="TeamsValidator", data_type="teams", is_valid=True)

        # Store metadata
        result.metadata["league_id"] = self.league_id
        result.metadata["year"] = self.year
        if self.division:
            result.metadata["division"] = self.division

        # Check not empty
        if not self._check_not_empty(df, result):
            return result

        result.metadata["row_count"] = len(df)

        # Check required columns
        if not self._check_required_columns(df, self.REQUIRED_COLUMNS, result):
            return result

        # Check no null values in critical columns
        self._check_no_null_values(df, {"Team Name"}, result)

        # Check data types
        expected_types = {
            "Team Name": "str",
            "Home": "str",
            "Convenor": "str",
            "Email": "str",
        }
        self._check_data_types(df, expected_types, result)

        # Check row count is reasonable
        self._check_row_count_range(df, self.EXPECTED_MIN_TEAMS, self.EXPECTED_MAX_TEAMS, result)

        # Check for duplicate team names
        if "Team Name" in df.columns:
            duplicates = df["Team Name"].duplicated().sum()
            if duplicates > 0:
                result.add_error("Team Name", f"Found {duplicates} duplicate team names")

        # Check email format (basic validation)
        if "Email" in df.columns:
            invalid_emails = (
                df[df["Email"].notna()]["Email"].apply(lambda x: "@" not in str(x) if pd.notna(x) else False).sum()
            )

            if invalid_emails > 0:
                result.add_warning("Email", f"{invalid_emails} emails missing @ symbol")

        # Check for empty convenor names
        if "Convenor" in df.columns:
            empty_convenors = df["Convenor"].isna().sum()
            if empty_convenors > 0:
                result.add_info("Convenor", f"{empty_convenors} teams have no convenor listed")

        self._log_result(result)
        return result
