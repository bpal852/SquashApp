"""
Configuration Management for HK Squash League Scraper.

This module provides environment-aware configuration with validation and type safety.
Supports development, testing, and production environments.

Usage:
    from config.settings import get_config

    config = get_config()
    print(config.BASE_URL)
    print(config.DIVISIONS)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


@dataclass
class DivisionConfig:
    """Configuration for a single division."""

    id: int
    day: str
    enabled: bool

    def __post_init__(self):
        """Validate division configuration."""
        valid_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if self.day not in valid_days:
            raise ConfigValidationError(f"Invalid day '{self.day}'. Must be one of {valid_days}")
        if not isinstance(self.id, int) or self.id <= 0:
            raise ConfigValidationError(f"Division id must be a positive integer, got {self.id}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DivisionConfig":
        """Create DivisionConfig from dictionary."""
        return cls(id=data["id"], day=data["day"], enabled=data["enabled"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"id": self.id, "day": self.day, "enabled": self.enabled}


class BaseConfig:
    """Base configuration with common settings."""

    # Project paths
    REPO_ROOT: Path = Path(__file__).resolve().parent.parent

    # API Configuration
    BASE_URL: str = "https://www.hksquash.org.hk/public/index.php/leagues"
    PAGES_ID: str = "25"

    # Season Configuration
    SEASON_YEAR: str = "2025-2026"

    # Request Configuration
    REQUEST_TIMEOUT: tuple = (10, 30)  # (connect timeout, read timeout)
    WAIT_TIME: int = 30  # seconds between requests
    RETRY_TOTAL: int = 5
    RETRY_BACKOFF_FACTOR: float = 1.5
    RETRY_STATUS_FORCELIST: list = [429, 500, 502, 503, 504]

    # Feature Flags
    ENABLE_VALIDATION: bool = True

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
    LOG_BACKUP_COUNT: int = 5
    LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"

    # Division Configuration
    # Format: "Division Name": {"id": int, "day": str, "enabled": bool}
    DIVISIONS: Dict[str, Dict[str, Any]] = {
        # Mondays
        "2": {"id": 473, "day": "Mon", "enabled": True},
        "6": {"id": 477, "day": "Mon", "enabled": True},
        "10": {"id": 482, "day": "Mon", "enabled": True},
        # Tuesdays
        "3": {"id": 474, "day": "Tue", "enabled": True},
        "4": {"id": 475, "day": "Tue", "enabled": True},
        "11": {"id": 483, "day": "Tue", "enabled": True},
        "L2": {"id": 496, "day": "Tue", "enabled": True},
        # Wednesdays
        "7": {"id": 478, "day": "Wed", "enabled": True},
        "9": {"id": 481, "day": "Wed", "enabled": True},
        "12": {"id": 484, "day": "Wed", "enabled": True},
        "M2": {"id": 492, "day": "Wed", "enabled": True},
        # Thursdays
        "Premier Main": {"id": 472, "day": "Thu", "enabled": True},
        "Premier Masters": {"id": 491, "day": "Thu", "enabled": True},
        "Premier Ladies": {"id": 495, "day": "Thu", "enabled": True},
        "M3": {"id": 493, "day": "Thu", "enabled": True},
        "M4": {"id": 494, "day": "Thu", "enabled": True},
        # Fridays
        "5": {"id": 476, "day": "Fri", "enabled": True},
        "8A": {"id": 479, "day": "Fri", "enabled": True},
        "8B": {"id": 480, "day": "Fri", "enabled": True},
        "13A": {"id": 485, "day": "Fri", "enabled": True},
        "13B": {"id": 486, "day": "Fri", "enabled": True},
        "13C": {"id": 487, "day": "Fri", "enabled": True},
        "L3": {"id": 497, "day": "Fri", "enabled": True},
        "L4": {"id": 498, "day": "Fri", "enabled": True},
        # Saturdays
        "14": {"id": 488, "day": "Sat", "enabled": True},
        "15A": {"id": 489, "day": "Sat", "enabled": True},
        "15B": {"id": 490, "day": "Sat", "enabled": True},
    }

    # Scraping Configuration
    MAX_RUBBERS: int = 5  # Maximum number of rubbers per match

    def __init__(self):
        """Initialize configuration and validate."""
        self.validate()

    def validate(self):
        """Validate configuration settings."""
        # Validate paths
        if not self.REPO_ROOT.exists():
            raise ConfigValidationError(f"REPO_ROOT does not exist: {self.REPO_ROOT}")

        # Validate URLs
        if not self.BASE_URL.startswith("http"):
            raise ConfigValidationError(f"BASE_URL must start with http/https: {self.BASE_URL}")

        # Validate timeouts
        if len(self.REQUEST_TIMEOUT) != 2:
            raise ConfigValidationError("REQUEST_TIMEOUT must be a tuple of (connect_timeout, read_timeout)")
        if any(t <= 0 for t in self.REQUEST_TIMEOUT):
            raise ConfigValidationError("REQUEST_TIMEOUT values must be positive")

        # Validate wait time
        if self.WAIT_TIME < 0:
            raise ConfigValidationError("WAIT_TIME must be non-negative")

        # Validate divisions
        if not self.DIVISIONS:
            raise ConfigValidationError("DIVISIONS cannot be empty")

        for name, config in self.DIVISIONS.items():
            try:
                DivisionConfig.from_dict(config)
            except Exception as e:
                raise ConfigValidationError(f"Invalid division '{name}': {e}")

    def get_enabled_divisions(self) -> Dict[str, int]:
        """Get dictionary of enabled divisions {name: id}."""
        return {name: meta["id"] for name, meta in self.DIVISIONS.items() if meta["enabled"]}

    def get_all_divisions(self) -> Dict[str, int]:
        """Get dictionary of all divisions {name: id}."""
        return {name: meta["id"] for name, meta in self.DIVISIONS.items()}

    def get_weekday_groups(self) -> Dict[str, Dict[str, int]]:
        """Get divisions grouped by weekday {day: {name: id}}."""
        weekday_groups = {}
        for name, meta in self.DIVISIONS.items():
            if meta["enabled"]:
                weekday_groups.setdefault(meta["day"], {})[name] = meta["id"]
        return weekday_groups

    def get_output_directories(self) -> Dict[str, str]:
        """Get all output directory paths."""
        year_path = self.REPO_ROOT / self.SEASON_YEAR
        return {
            "summary_df": str(year_path / "summary_df"),
            "teams_df": str(year_path / "teams_df"),
            "schedules_df": str(year_path / "schedules_df"),
            "ranking_df": str(year_path / "ranking_df"),
            "players_df": str(year_path / "players_df"),
            "summarized_player_tables": str(year_path / "summarized_player_tables"),
            "unbeaten_players": str(year_path / "unbeaten_players"),
            "played_every_game": str(year_path / "played_every_game"),
            "detailed_league_tables": str(year_path / "detailed_league_tables"),
            "awaiting_results": str(year_path / "awaiting_results"),
            "home_away_data": str(year_path / "home_away_data"),
            "team_win_percentage_breakdown_home": str(year_path / "team_win_percentage_breakdown" / "Home"),
            "team_win_percentage_breakdown_away": str(year_path / "team_win_percentage_breakdown" / "Away"),
            "team_win_percentage_breakdown_delta": str(year_path / "team_win_percentage_breakdown" / "Delta"),
            "team_win_percentage_breakdown_overall": str(year_path / "team_win_percentage_breakdown" / "Overall"),
            "simulated_tables": str(year_path / "simulated_tables"),
            "simulated_fixtures": str(year_path / "simulated_fixtures"),
            "remaining_fixtures": str(year_path / "remaining_fixtures"),
            "neutral_fixtures": str(year_path / "neutral_fixtures"),
            "results_df": str(year_path / "results_df"),
            "logs": str(year_path / "logs"),
        }

    def get_log_file_path(self) -> Path:
        """Get the log file path for this season."""
        return self.REPO_ROOT / self.SEASON_YEAR / "logs" / f"{self.SEASON_YEAR}_log.txt"

    def build_url(self, path: str, league_id: Optional[int] = None) -> str:
        """Build a complete URL for the API."""
        url = f"{self.BASE_URL}/{path}/pages_id/{self.PAGES_ID}"
        if league_id is not None:
            url += f"/league_id/{league_id}"
        return url


class DevelopmentConfig(BaseConfig):
    """Development environment configuration."""

    # Override for development
    WAIT_TIME: int = 5  # Faster for development
    LOG_LEVEL: str = "DEBUG"

    # Enable only a subset of divisions for faster testing
    DIVISIONS: Dict[str, Dict[str, Any]] = {
        "2": {"id": 473, "day": "Mon", "enabled": True},
        "6": {"id": 477, "day": "Mon", "enabled": True},
        "10": {"id": 482, "day": "Mon", "enabled": True},
        # All other divisions disabled in dev mode
        "3": {"id": 474, "day": "Tue", "enabled": False},
        "4": {"id": 475, "day": "Tue", "enabled": False},
        "11": {"id": 483, "day": "Tue", "enabled": False},
        "L2": {"id": 496, "day": "Tue", "enabled": False},
        "7": {"id": 478, "day": "Wed", "enabled": False},
        "9": {"id": 481, "day": "Wed", "enabled": False},
        "12": {"id": 484, "day": "Wed", "enabled": False},
        "M2": {"id": 492, "day": "Wed", "enabled": False},
        "Premier Main": {"id": 472, "day": "Thu", "enabled": False},
        "Premier Masters": {"id": 491, "day": "Thu", "enabled": False},
        "Premier Ladies": {"id": 495, "day": "Thu", "enabled": False},
        "M3": {"id": 493, "day": "Thu", "enabled": False},
        "M4": {"id": 494, "day": "Thu", "enabled": False},
        "5": {"id": 476, "day": "Fri", "enabled": False},
        "8A": {"id": 479, "day": "Fri", "enabled": False},
        "8B": {"id": 480, "day": "Fri", "enabled": False},
        "13A": {"id": 485, "day": "Fri", "enabled": False},
        "13B": {"id": 486, "day": "Fri", "enabled": False},
        "13C": {"id": 487, "day": "Fri", "enabled": False},
        "L3": {"id": 497, "day": "Fri", "enabled": False},
        "L4": {"id": 498, "day": "Fri", "enabled": False},
        "14": {"id": 488, "day": "Sat", "enabled": False},
        "15A": {"id": 489, "day": "Sat", "enabled": False},
        "15B": {"id": 490, "day": "Sat", "enabled": False},
    }


class TestingConfig(BaseConfig):
    """Testing environment configuration."""

    # Override for testing
    WAIT_TIME: int = 1  # Minimal delays in tests (0 not allowed by validation)
    ENABLE_VALIDATION: bool = True  # Always validate in tests
    LOG_LEVEL: str = "WARNING"  # Less verbose for tests

    # Minimal divisions for fast tests
    DIVISIONS: Dict[str, Dict[str, Any]] = {
        "2": {"id": 473, "day": "Mon", "enabled": True},
        "6": {"id": 477, "day": "Mon", "enabled": True},
    }


class ProductionConfig(BaseConfig):
    """Production environment configuration - all divisions enabled."""

    # Production defaults are inherited from BaseConfig
    pass


# Configuration registry
_config_map = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}

# Singleton instance
_config_instance: Optional[BaseConfig] = None


def get_config(env: Optional[str] = None) -> BaseConfig:
    """
    Get configuration instance for the specified environment.

    Args:
        env: Environment name ('development', 'testing', 'production').
             If None, reads from SQUASH_ENV environment variable.
             Defaults to 'production' if not set.

    Returns:
        Configuration instance.

    Raises:
        ConfigValidationError: If configuration is invalid.

    Example:
        >>> config = get_config('development')
        >>> print(config.BASE_URL)
        >>> print(len(config.get_enabled_divisions()))
    """
    global _config_instance

    if _config_instance is None:
        # Determine environment
        if env is None:
            env = os.getenv("SQUASH_ENV", "production")

        env = env.lower()
        if env not in _config_map:
            raise ConfigValidationError(f"Invalid environment '{env}'. Must be one of: {list(_config_map.keys())}")

        # Create and cache instance
        _config_instance = _config_map[env]()

    return _config_instance


def reset_config():
    """Reset configuration singleton. Useful for testing."""
    global _config_instance
    _config_instance = None


def print_config_summary(config: BaseConfig):
    """Print a summary of the current configuration."""
    enabled_divisions = config.get_enabled_divisions()

    print("=" * 70)
    print("🔧 CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Environment: {config.__class__.__name__}")
    print(f"Season Year: {config.SEASON_YEAR}")
    print(f"Base URL: {config.BASE_URL}")
    print(f"Validation: {'✅ Enabled' if config.ENABLE_VALIDATION else '❌ Disabled'}")
    print(f"Wait Time: {config.WAIT_TIME}s between requests")
    print(f"Request Timeout: {config.REQUEST_TIMEOUT[0]}s connect, {config.REQUEST_TIMEOUT[1]}s read")
    print(f"Log Level: {config.LOG_LEVEL}")
    print("-" * 70)
    print(f"📊 Enabled Divisions ({len(enabled_divisions)}):")

    weekday_groups = config.get_weekday_groups()
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        if day in weekday_groups:
            divisions = ", ".join(weekday_groups[day].keys())
            print(f"  {day}: {divisions}")

    print("-" * 70)
    total_divisions = len(config.DIVISIONS)
    runtime_estimate = f"~{len(enabled_divisions) * 2}-{len(enabled_divisions) * 3} minutes"
    if len(enabled_divisions) == total_divisions:
        runtime_estimate = "~60-90 minutes"
    print(f"⏱️  Estimated Runtime: {runtime_estimate}")
    print("=" * 70)


if __name__ == "__main__":
    # Test configuration
    print("Testing Development Config:")
    reset_config()
    dev_config = get_config("development")
    print_config_summary(dev_config)

    print("\n")

    print("Testing Production Config:")
    reset_config()
    prod_config = get_config("production")
    print_config_summary(prod_config)
