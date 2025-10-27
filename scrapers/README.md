# Scrapers Package

This package contains modular scraper classes for extracting data from the Hong Kong Squash website.

## Architecture

All scrapers inherit from `BaseScraper`, which provides common functionality:
- Session management with retry logic (3 attempts with exponential backoff)
- Timeout handling (30 seconds per request)
- URL construction
- Error logging
- Page fetching with automatic retries

## Scrapers

### BaseScraper (`base.py`)

**Base class providing common scraper functionality.**

```python
from scrapers import BaseScraper

scraper = BaseScraper(session=requests.Session())
soup = scraper.fetch_page("D00473", "2025-2026", "teams")
```

**Key Methods:**
- `fetch_page(league_id, year, page_type)` - Fetches and parses HTML page
- `build_url(league_id, year, page_type)` - Constructs URL for given parameters
- `_log_info()`, `_log_error()` - Logging helpers

**Features:**
- 3 retry attempts with exponential backoff (1s, 2s, 4s delays)
- 30 second timeout per request
- Automatic logging of successes and failures

---

### TeamsScraper (`teams.py`)

**Extracts team information including rosters.**

```python
from scrapers import TeamsScraper

scraper = TeamsScraper(session=session)
df = scraper.scrape("D00473", "2025-2026")
```

**Returns:** DataFrame with columns:
- `Team Name` (str)
- `Home` (str)
- `Convenor` (str)
- `Email` (str)

**Example Output:**
```
Team Name                    Home                  Convenor        Email
Kowloon Cricket Club 2       Kowloon Cricket Club  John Smith     john@email.com
```

---

### TeamSummaryScraper (`summary.py`)

**Extracts match summary statistics.**

```python
from scrapers import TeamSummaryScraper

scraper = TeamSummaryScraper(session=session)
df = scraper.scrape("D00473", "2025-2026")
```

**Returns:** DataFrame with columns:
- `Team` (str)
- `Played` (int)
- `Won` (int)
- `Lost` (int)
- `Points` (int)

**Special Features:**
- Handles both "team_summery" (misspelled) and "team_summary" URL variants
- Automatically tries fallback spelling if first attempt fails
- Filters out header rows containing "Team"

**Example Output:**
```
Team                        Played  Won  Lost  Points
Kowloon Cricket Club 2      2       2    0     10
Hong Kong Football Club 2A  2       1    1     5
```

---

### SchedulesAndResultsScraper (`schedules.py`)

**Extracts fixtures and results.**

```python
from scrapers import SchedulesAndResultsScraper

scraper = SchedulesAndResultsScraper(session=session)
df = scraper.scrape("D00473", "2025-2026")
```

**Returns:** DataFrame with columns:
- `Home Team` (str)
- `vs` (str) - literal "vs"
- `Away Team` (str)
- `Venue` (str)
- `Time` (str)
- `Result` (str) - empty string if not yet played
- `Match Week` (int)
- `Date` (datetime)

**Helper Methods:**
- `_extract_match_week_and_date()` - Parses week headers
- `_extract_row_data()` - Extracts data from table rows
- `_create_dataframe()` - Constructs final DataFrame

**Example Output:**
```
Home Team           vs  Away Team               Venue      Time   Result  Match Week  Date
Kowloon CC 2        vs  Hong Kong FC 2A        HKCC       19:30  3-2     1          2025-10-02
```

---

### RankingScraper (`ranking.py`)

**Extracts player rankings and generates statistics.**

```python
from scrapers import RankingScraper

scraper = RankingScraper(session=session)
ranking_df, summarized_df, unbeaten_list, filtered_df = scraper.scrape("D00473", "2025-2026")
```

**Returns:** Tuple of (ranking_df, summarized_df, unbeaten_list, ranking_df_filtered)

**ranking_df columns:**
- `Player` (str)
- `Games` (int)
- `Won` (int)
- `Lost` (int)
- `W %` (float) - Win percentage
- `5-0` through `0-5` (int) - Score distribution
- `Team` (str)

**Features:**
- Filters out players with < 5 games played
- Generates team summary statistics
- Creates list of unbeaten players
- Handles "NO DATA" scenarios gracefully

**Example Usage:**
```python
# Get all ranking data
ranking_df, summarized_df, unbeaten_list, filtered_df = scraper.scrape("D00473", "2025-2026")

# Access unbeaten players
print(unbeaten_list)  # ['John Smith (Team A)', 'Jane Doe (Team B)']

# Get filtered rankings (≥5 games only)
print(filtered_df)
```

---

### PlayersScraper (`players.py`)

**Extracts player rosters by team.**

```python
from scrapers import PlayersScraper

scraper = PlayersScraper(session=session)
df = scraper.scrape("D00473", "2025-2026")
```

**Returns:** DataFrame with columns:
- `Order` (int)
- `Player` (str)
- `HKS No.` (int)
- `Ranking` (int)
- `Points` (float)
- `Team` (str)

**Features:**
- Handles "NO DATA" teams (skips gracefully)
- Validates row format (must have 5 columns)
- Converts numeric fields to proper types
- Skips malformed rows with logging

**Example Output:**
```
Order  Player        HKS No.  Ranking  Points  Team
1      John Smith    12345    150      3.5     Kowloon Cricket Club 2
2      Jane Doe      23456    200      2.8     Kowloon Cricket Club 2
```

---

## Usage in Main Script

```python
import requests
from scrapers import (
    TeamsScraper,
    TeamSummaryScraper, 
    SchedulesAndResultsScraper,
    RankingScraper,
    PlayersScraper
)

# Create shared session
SESSION = requests.Session()

# Use scrapers
teams_df = TeamsScraper(session=SESSION).scrape(league_id, year)
summary_df = TeamSummaryScraper(session=SESSION).scrape(league_id, year)
schedules_df = SchedulesAndResultsScraper(session=SESSION).scrape(league_id, year)
ranking_data = RankingScraper(session=SESSION).scrape(league_id, year)
players_df = PlayersScraper(session=SESSION).scrape(league_id, year)
```

## Testing

All scrapers have comprehensive unit tests in `tests/test_scrapers.py` with mocked HTTP responses. See the [tests README](../tests/README.md) for details.

## Error Handling

All scrapers implement robust error handling:
- **HTTP Errors**: Logged with details, raises HTTPError
- **Empty Pages**: Returns empty DataFrame or appropriate empty values
- **Malformed HTML**: Logs warnings and skips bad data
- **Timeouts**: Retries up to 3 times with exponential backoff
- **NO DATA**: Handles gracefully (common for certain divisions)

## Logging

All scrapers use Python's logging module:
- `INFO`: Successful operations
- `WARNING`: Recoverable issues (bad data, empty pages)
- `ERROR`: Failed operations requiring attention

Configure logging in your main script:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Design Principles

1. **Single Responsibility**: Each scraper handles one type of data
2. **DRY (Don't Repeat Yourself)**: Common functionality in BaseScraper
3. **Testability**: All scrapers can be tested with mocked responses
4. **Robustness**: Graceful error handling for all edge cases
5. **Type Safety**: Proper data type conversions (int, float, datetime)
