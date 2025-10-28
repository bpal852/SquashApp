# Configuration Management

This package provides environment-aware configuration management for the HK Squash League Scraper.

## Features

- **Environment-aware**: Separate configurations for development, testing, and production
- **Type-safe**: Using dataclasses and type hints
- **Validated**: Automatic validation on initialization
- **Centralized**: Single source of truth for all configuration
- **Documented**: Clear examples and usage patterns

## Usage

### Basic Usage

```python
from config import get_config

# Get configuration (defaults to production)
config = get_config()

# Use configuration
print(config.BASE_URL)
print(config.SEASON_YEAR)
print(config.get_enabled_divisions())
```

### Environment Selection

You can specify the environment in two ways:

**1. Programmatically:**

```python
from config import get_config

# Development environment (limited divisions)
dev_config = get_config('development')

# Testing environment (minimal divisions)
test_config = get_config('testing')

# Production environment (all divisions)
prod_config = get_config('production')
```

**2. Environment Variable:**

```bash
# Windows PowerShell
$env:SQUASH_ENV = "development"
python main_2025_26.py

# Linux/Mac
export SQUASH_ENV=development
python main_2025_26.py
```

### Configuration Summary

Print a summary of the current configuration:

```python
from config import get_config, print_config_summary

config = get_config('development')
print_config_summary(config)
```

Output:
```
======================================================================
🔧 CONFIGURATION SUMMARY
======================================================================
Environment: DevelopmentConfig
Season Year: 2025-2026
Base URL: https://www.hksquash.org.hk/public/index.php/leagues
Validation: ✅ Enabled
Wait Time: 5s between requests
Request Timeout: 10s connect, 30s read
Log Level: DEBUG
----------------------------------------------------------------------
📊 Enabled Divisions (3):
  Mon: Premier 1, Premier 2, 1
----------------------------------------------------------------------
⏱️  Estimated Runtime: ~6-9 minutes
======================================================================
```

## Environments

### Development

- **Purpose**: Local development and testing
- **Divisions**: Only 3 divisions enabled (Premier 1, Premier 2, Division 1)
- **Wait Time**: 5 seconds (faster than production)
- **Log Level**: DEBUG (verbose output)
- **Runtime**: ~6-9 minutes

### Testing

- **Purpose**: Automated unit and integration tests
- **Divisions**: Only 2 divisions enabled (minimal set)
- **Wait Time**: 0 seconds (no delays)
- **Log Level**: WARNING (minimal output)
- **Runtime**: ~4-6 minutes

### Production

- **Purpose**: Full scraping runs
- **Divisions**: All 35+ divisions enabled
- **Wait Time**: 30 seconds (respectful of server)
- **Log Level**: INFO
- **Runtime**: ~60-90 minutes

## Configuration Options

### API Configuration

```python
config.BASE_URL              # Base URL for HK Squash website
config.PAGES_ID              # Pages ID parameter
config.REQUEST_TIMEOUT       # (connect_timeout, read_timeout)
config.WAIT_TIME             # Seconds between requests
config.RETRY_TOTAL           # Number of retries
config.RETRY_BACKOFF_FACTOR  # Exponential backoff factor
```

### Season Configuration

```python
config.SEASON_YEAR           # Current season (e.g., "2025-2026")
```

### Feature Flags

```python
config.ENABLE_VALIDATION     # Enable data validation
```

### Division Management

```python
# Get all enabled divisions
enabled = config.get_enabled_divisions()
# Returns: {"Premier 1": 463, "Premier 2": 464, ...}

# Get all divisions (including disabled)
all_divs = config.get_all_divisions()

# Get divisions grouped by weekday
weekdays = config.get_weekday_groups()
# Returns: {"Mon": {"Premier 1": 463, ...}, "Tue": {...}, ...}
```

### Output Directories

```python
# Get all output directory paths
dirs = config.get_output_directories()
# Returns: {
#     'summary_df': '/path/to/2025-2026/summary_df',
#     'teams_df': '/path/to/2025-2026/teams_df',
#     ...
# }
```

### URL Building

```python
# Build API URLs
url = config.build_url('league-team', league_id=463)
# Returns: "https://www.hksquash.org.hk/public/index.php/leagues/league-team/pages_id/25/league_id/463"
```

## Division Configuration

Each division has the following structure:

```python
{
    "Division Name": {
        "id": 463,              # Division ID on HK Squash website
        "day": "Mon",           # Match day (Mon/Tue/Wed/Thu/Fri/Sat/Sun)
        "enabled": True         # Whether to scrape this division
    }
}
```

Example:

```python
config.DIVISIONS = {
    "Premier 1": {"id": 463, "day": "Mon", "enabled": True},
    "Premier 2": {"id": 464, "day": "Mon", "enabled": True},
    "1": {"id": 465, "day": "Mon", "enabled": True},
    # ... more divisions
}
```

## Validation

Configuration is automatically validated on initialization. Common validation checks:

- ✅ REPO_ROOT exists
- ✅ BASE_URL starts with http/https
- ✅ REQUEST_TIMEOUT is a tuple of 2 positive numbers
- ✅ WAIT_TIME is positive
- ✅ All divisions have valid day names
- ✅ All division IDs are positive integers

If validation fails, a `ConfigValidationError` is raised:

```python
from config import ConfigValidationError

try:
    config = get_config()
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

## Testing

For testing, always reset the configuration singleton:

```python
from config import get_config, reset_config

def test_something():
    reset_config()  # Clear cached config
    config = get_config('testing')
    # ... test code
```

## Adding New Configuration

To add new configuration options:

1. Add to `BaseConfig` class in `settings.py`:

```python
class BaseConfig:
    # ... existing config
    
    # New configuration
    MY_NEW_SETTING: str = "default_value"
```

2. Override in environment-specific classes if needed:

```python
class DevelopmentConfig(BaseConfig):
    MY_NEW_SETTING: str = "dev_value"
```

3. Add validation if necessary:

```python
def validate(self):
    super().validate()
    
    # Custom validation
    if not self.MY_NEW_SETTING:
        raise ConfigValidationError("MY_NEW_SETTING cannot be empty")
```

## Best Practices

1. **Always use `get_config()`**: Never instantiate config classes directly
2. **Use environment variables**: Set `SQUASH_ENV` for different environments
3. **Validate early**: Let configuration validation catch errors before scraping starts
4. **Test with different environments**: Ensure your code works in dev, test, and prod
5. **Document new settings**: Update this README when adding new configuration

## Migration from Hardcoded Values

Old code:
```python
BASE = "https://www.hksquash.org.hk/public/index.php/leagues"
wait_time = 30
DIVISIONS = {...}
```

New code:
```python
from config import get_config

config = get_config()
base_url = config.BASE_URL
wait_time = config.WAIT_TIME
divisions = config.DIVISIONS
```

## Troubleshooting

### Configuration not loading

Make sure the `config` directory is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

### Environment variable not working

Check that `SQUASH_ENV` is set correctly:

```python
import os
print(os.getenv('SQUASH_ENV'))  # Should print: development/testing/production
```

### Validation errors

Run config validation standalone:

```bash
python config/settings.py
```

This will print configuration summaries for development and production environments.
