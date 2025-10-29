"""
Unit tests for the configuration management system.

Tests configuration loading, validation, environment switching, and helper methods.
"""

import os
import pytest
from pathlib import Path

from config import (
    BaseConfig,
    DevelopmentConfig,
    TestingConfig,
    ProductionConfig,
    get_config,
    reset_config,
    ConfigValidationError,
    DivisionConfig,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_config_after_test():
    """Reset config singleton after each test."""
    yield
    reset_config()


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables."""
    def _set_env(env_name):
        monkeypatch.setenv('SQUASH_ENV', env_name)
    return _set_env


# ============================================================================
# DivisionConfig Tests
# ============================================================================

class TestDivisionConfig:
    """Tests for DivisionConfig dataclass."""
    
    def test_valid_division_config(self):
        """Test creating a valid division config."""
        config = DivisionConfig(id=463, day="Mon", enabled=True)
        assert config.id == 463
        assert config.day == "Mon"
        assert config.enabled is True
    
    def test_division_config_from_dict(self):
        """Test creating division config from dictionary."""
        data = {"id": 463, "day": "Mon", "enabled": True}
        config = DivisionConfig.from_dict(data)
        assert config.id == 463
        assert config.day == "Mon"
        assert config.enabled is True
    
    def test_division_config_to_dict(self):
        """Test converting division config to dictionary."""
        config = DivisionConfig(id=463, day="Mon", enabled=True)
        data = config.to_dict()
        assert data == {"id": 463, "day": "Mon", "enabled": True}
    
    def test_invalid_day(self):
        """Test that invalid day raises error."""
        with pytest.raises(ConfigValidationError):
            DivisionConfig(id=463, day="InvalidDay", enabled=True)
    
    def test_invalid_id_negative(self):
        """Test that negative ID raises error."""
        with pytest.raises(ConfigValidationError):
            DivisionConfig(id=-1, day="Mon", enabled=True)
    
    def test_invalid_id_zero(self):
        """Test that zero ID raises error."""
        with pytest.raises(ConfigValidationError):
            DivisionConfig(id=0, day="Mon", enabled=True)


# ============================================================================
# BaseConfig Tests
# ============================================================================

class TestBaseConfig:
    """Tests for BaseConfig class."""
    
    def test_base_config_initialization(self):
        """Test BaseConfig initializes correctly."""
        config = BaseConfig()
        assert config.BASE_URL == "https://www.hksquash.org.hk/public/index.php/leagues"
        assert config.PAGES_ID == "25"
        assert config.SEASON_YEAR == "2025-2026"
        assert config.WAIT_TIME == 30
        assert config.ENABLE_VALIDATION is True
    
    def test_base_config_has_divisions(self):
        """Test BaseConfig has divisions defined."""
        config = BaseConfig()
        assert len(config.DIVISIONS) > 0
        assert "2" in config.DIVISIONS
        assert config.DIVISIONS["2"]["id"] == 473
    
    def test_get_enabled_divisions(self):
        """Test getting enabled divisions."""
        config = BaseConfig()
        enabled = config.get_enabled_divisions()
        assert isinstance(enabled, dict)
        assert len(enabled) > 0
        assert all(isinstance(v, int) for v in enabled.values())
    
    def test_get_all_divisions(self):
        """Test getting all divisions."""
        config = BaseConfig()
        all_divs = config.get_all_divisions()
        assert isinstance(all_divs, dict)
        assert len(all_divs) == len(config.DIVISIONS)
    
    def test_get_weekday_groups(self):
        """Test getting divisions grouped by weekday."""
        config = BaseConfig()
        weekdays = config.get_weekday_groups()
        assert isinstance(weekdays, dict)
        assert "Mon" in weekdays
        assert isinstance(weekdays["Mon"], dict)
    
    def test_get_output_directories(self):
        """Test getting output directory paths."""
        config = BaseConfig()
        dirs = config.get_output_directories()
        assert isinstance(dirs, dict)
        assert 'summary_df' in dirs
        assert 'teams_df' in dirs
        assert 'schedules_df' in dirs
        assert all(isinstance(v, str) for v in dirs.values())
    
    def test_get_log_file_path(self):
        """Test getting log file path."""
        config = BaseConfig()
        log_path = config.get_log_file_path()
        assert isinstance(log_path, Path)
        assert log_path.name.endswith("_log.txt")
    
    def test_build_url_without_league_id(self):
        """Test building URL without league ID."""
        config = BaseConfig()
        url = config.build_url('league-team')
        assert url.startswith(config.BASE_URL)
        assert 'pages_id' in url
        assert config.PAGES_ID in url
        assert 'league_id' not in url
    
    def test_build_url_with_league_id(self):
        """Test building URL with league ID."""
        config = BaseConfig()
        url = config.build_url('league-team', league_id=463)
        assert url.startswith(config.BASE_URL)
        assert 'pages_id' in url
        assert 'league_id/463' in url


# ============================================================================
# Environment-Specific Config Tests
# ============================================================================

class TestDevelopmentConfig:
    """Tests for DevelopmentConfig."""
    
    def test_development_config(self):
        """Test development config has correct settings."""
        config = DevelopmentConfig()
        assert config.WAIT_TIME == 5  # Faster than production
        assert config.LOG_LEVEL == "DEBUG"
        
        # Should have limited divisions enabled
        enabled = config.get_enabled_divisions()
        assert len(enabled) == 3
        assert "2" in enabled
        assert "6" in enabled
        assert "10" in enabled


class TestTestingConfig:
    """Tests for TestingConfig."""
    
    def test_testing_config(self):
        """Test testing config has correct settings."""
        config = TestingConfig()
        assert config.WAIT_TIME == 1  # Minimal delays
        assert config.LOG_LEVEL == "WARNING"
        assert config.ENABLE_VALIDATION is True
        
        # Should have minimal divisions
        enabled = config.get_enabled_divisions()
        assert len(enabled) == 2


class TestProductionConfig:
    """Tests for ProductionConfig."""
    
    def test_production_config(self):
        """Test production config has correct settings."""
        config = ProductionConfig()
        assert config.WAIT_TIME == 30
        assert config.LOG_LEVEL == "INFO"
        
        # Should have all divisions enabled
        enabled = config.get_enabled_divisions()
        assert len(enabled) > 30


# ============================================================================
# get_config() Function Tests
# ============================================================================

class TestGetConfig:
    """Tests for get_config() function."""
    
    def test_get_config_development(self):
        """Test getting development config."""
        config = get_config('development')
        assert isinstance(config, DevelopmentConfig)
        assert config.WAIT_TIME == 5
    
    def test_get_config_testing(self):
        """Test getting testing config."""
        config = get_config('testing')
        assert isinstance(config, TestingConfig)
        assert config.WAIT_TIME == 1
    
    def test_get_config_production(self):
        """Test getting production config."""
        config = get_config('production')
        assert isinstance(config, ProductionConfig)
        assert config.WAIT_TIME == 30
    
    def test_get_config_default_is_production(self):
        """Test that default config is production."""
        config = get_config()
        assert isinstance(config, ProductionConfig)
    
    def test_get_config_from_env_variable(self, mock_env):
        """Test getting config from environment variable."""
        mock_env('development')
        config = get_config()
        assert isinstance(config, DevelopmentConfig)
    
    def test_get_config_invalid_environment(self):
        """Test that invalid environment raises error."""
        with pytest.raises(ConfigValidationError):
            get_config('invalid_env')
    
    def test_get_config_singleton(self):
        """Test that get_config() returns same instance."""
        config1 = get_config('development')
        config2 = get_config('development')
        assert config1 is config2
    
    def test_reset_config(self):
        """Test that reset_config() clears singleton."""
        config1 = get_config('development')
        reset_config()
        config2 = get_config('production')
        assert config1 is not config2
        assert isinstance(config1, DevelopmentConfig)
        assert isinstance(config2, ProductionConfig)


# ============================================================================
# Validation Tests
# ============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_validation_runs_on_init(self):
        """Test that validation runs during initialization."""
        # This should not raise an error
        config = BaseConfig()
        assert config is not None
    
    def test_repo_root_exists(self):
        """Test that REPO_ROOT validation works."""
        config = BaseConfig()
        assert config.REPO_ROOT.exists()
    
    def test_base_url_format(self):
        """Test BASE_URL validation."""
        config = BaseConfig()
        assert config.BASE_URL.startswith("http")
    
    def test_request_timeout_format(self):
        """Test REQUEST_TIMEOUT validation."""
        config = BaseConfig()
        assert len(config.REQUEST_TIMEOUT) == 2
        assert all(t > 0 for t in config.REQUEST_TIMEOUT)
    
    def test_divisions_not_empty(self):
        """Test that divisions dict is not empty."""
        config = BaseConfig()
        assert len(config.DIVISIONS) > 0
    
    def test_all_divisions_valid(self):
        """Test that all division configs are valid."""
        config = BaseConfig()
        for name, div_config in config.DIVISIONS.items():
            # Should not raise error
            DivisionConfig.from_dict(div_config)


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_config_works_with_real_paths(self):
        """Test that config works with real filesystem paths."""
        config = get_config('production')
        dirs = config.get_output_directories()
        
        # Check that paths are constructed correctly
        for key, path in dirs.items():
            assert isinstance(path, str)
            assert config.SEASON_YEAR in path
    
    def test_division_grouping_complete(self):
        """Test that all enabled divisions appear in weekday groups."""
        config = get_config('production')
        enabled = config.get_enabled_divisions()
        weekday_groups = config.get_weekday_groups()
        
        # Count divisions in weekday groups
        total_in_groups = sum(len(divs) for divs in weekday_groups.values())
        assert total_in_groups == len(enabled)
    
    def test_url_building_comprehensive(self):
        """Test URL building with various parameters."""
        config = get_config()
        
        # Without league_id
        url1 = config.build_url('league-team')
        assert 'league-team' in url1
        assert 'pages_id/25' in url1
        
        # With league_id
        url2 = config.build_url('league-team', league_id=463)
        assert 'league_id/463' in url2
        
        # Different path
        url3 = config.build_url('ranking', league_id=464)
        assert 'ranking' in url3
        assert 'league_id/464' in url3
    
    def test_different_environments_have_different_settings(self):
        """Test that different environments have distinct settings."""
        reset_config()
        dev = get_config('development')
        
        reset_config()
        test = get_config('testing')
        
        reset_config()
        prod = get_config('production')
        
        # Wait times should be different
        assert dev.WAIT_TIME < prod.WAIT_TIME
        assert test.WAIT_TIME < dev.WAIT_TIME
        
        # Division counts should be different
        assert len(dev.get_enabled_divisions()) < len(prod.get_enabled_divisions())
        assert len(test.get_enabled_divisions()) < len(dev.get_enabled_divisions())


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_case_insensitive_environment(self):
        """Test that environment names are case-insensitive."""
        reset_config()
        config1 = get_config('DEVELOPMENT')
        reset_config()
        config2 = get_config('Development')
        reset_config()
        config3 = get_config('development')
        
        assert type(config1) == type(config2) == type(config3)
    
    def test_empty_divisions_raises_error(self):
        """Test that empty divisions dict raises error."""
        class EmptyDivisionsConfig(BaseConfig):
            DIVISIONS = {}
        
        with pytest.raises(ConfigValidationError):
            EmptyDivisionsConfig()
    
    def test_invalid_timeout_raises_error(self):
        """Test that invalid timeout raises error."""
        class InvalidTimeoutConfig(BaseConfig):
            REQUEST_TIMEOUT = (10,)  # Only one value
        
        with pytest.raises(ConfigValidationError):
            InvalidTimeoutConfig()
    
    def test_negative_wait_time_raises_error(self):
        """Test that negative wait time raises error."""
        class NegativeWaitTimeConfig(BaseConfig):
            WAIT_TIME = -5
        
        with pytest.raises(ConfigValidationError):
            NegativeWaitTimeConfig()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
