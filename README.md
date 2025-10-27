# SquashApp

A comprehensive data scraping and analysis tool for Hong Kong Squash League data, built with Python, pandas, and Streamlit.

## Overview

SquashApp scrapes match results, player statistics, and team standings from the Hong Kong Squash website, processes the data, and provides interactive visualizations through a Streamlit dashboard.

## Features

- **Web Scraping**: Modular scrapers for teams, schedules, rankings, and player data
- **Data Processing**: Robust parsing of match results and player statistics
- **Analysis**: Home/away performance, win percentages, player rankings
- **Visualization**: Interactive Streamlit dashboard with multiple views
- **Historical Data**: Support for multiple seasons (2016-present)
- **Testing Mode**: Fast development mode (2 divisions vs 28)

## Project Structure

```
SquashApp/
├── scrapers/               # Modular web scraping package
│   ├── base.py            # BaseScraper with retry logic
│   ├── teams.py           # Team information scraper
│   ├── summary.py         # Match summary scraper
│   ├── schedules.py       # Fixtures and results scraper
│   ├── ranking.py         # Player rankings scraper
│   └── players.py         # Player details scraper
├── tests/                 # Comprehensive test suite
│   ├── test_parsers.py    # 58 parser tests
│   └── test_scrapers.py   # 20 scraper tests (mocked HTTP)
├── config/                # Division configurations
├── previous_seasons/      # Historical season data
├── 2025-2026/            # Current season data
├── parsers.py            # Pure parsing functions
├── main_2025_26.py       # Main orchestration script
└── app.py                # Streamlit dashboard

```

## Quick Start

### Installation

```powershell
# Clone repository
git clone https://github.com/bpal852/SquashApp.git
cd SquashApp

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Scraper

```powershell
# Full scrape (all 28 divisions, 60-90 minutes)
python main_2025_26.py

# Testing mode (2 divisions, 4-6 minutes)
# Set TESTING_MODE = True in main_2025_26.py
python main_2025_26.py
```

### Running the Dashboard

```powershell
streamlit run app.py
```

### Running Tests

```powershell
# Run all tests (78 tests, ~35 seconds)
pytest tests/ -v

# Run parser tests only (58 tests)
pytest tests/test_parsers.py -v

# Run scraper tests only (20 tests)
pytest tests/test_scrapers.py -v

# Run with coverage
pytest tests/ --cov=scrapers --cov=parsers --cov-report=html
```

## Architecture

### Scrapers Package

All scrapers inherit from `BaseScraper`, which provides:
- Session management with connection pooling
- Retry logic (3 attempts with exponential backoff)
- Timeout handling (30 seconds)
- Automatic logging
- URL construction

See [scrapers/README.md](scrapers/README.md) for detailed documentation.

### Parser Functions

Pure functions in `parsers.py` handle data transformation:
- `parse_result()` - Parse match scores (e.g., "3-2", "WO", "CR")
- `count_games_won()` - Count games from rubber results
- `normalize_rubber()` - Format rubber results consistently
- `determine_winner()` - Calculate match winner
- `format_rubbers()` - Format rubber lists
- `parse_home_away_scores()` - Extract home/away scores
- `extract_match_details()` - Parse match metadata
- `extract_match_result()` - Parse result strings

All functions have comprehensive unit tests with 100% coverage.

### Testing Strategy

**Parser Tests (58 tests)**
- Direct function calls with various inputs
- Normal cases, edge cases, error conditions
- Fast execution (milliseconds)

**Scraper Tests (20 tests)**
- Mocked HTTP responses using `unittest.mock`
- No network dependencies
- Controlled test scenarios
- Tests all edge cases (404, empty pages, malformed HTML)
- Fast execution (~30 seconds total)

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Data Flow

```
1. Scrape → 2. Parse → 3. Transform → 4. Analyze → 5. Visualize

1. Scrapers fetch HTML from HK Squash website
2. Parsers extract structured data from HTML
3. Data transformations (merge, filter, aggregate)
4. Statistical analysis (rankings, win %, trends)
5. Streamlit dashboard displays results
```

## Key Components

### main_2025_26.py

Main orchestration script that:
1. Reads division configuration
2. Scrapes data for each division
3. Processes and transforms data
4. Saves results to CSV files
5. Generates analysis outputs

**TESTING_MODE**: Set to `True` for fast development (scrapes only 2 divisions)

### app.py

Streamlit dashboard providing:
- **Team Stats**: Detailed league tables, match results, team analysis
- **Player Stats**: Individual player performance, rankings
- **Historical Data**: Multi-season comparisons
- Interactive filters and visualizations

### parsers.py

Pure parsing functions with no side effects:
- Type-safe conversions
- Robust error handling
- Comprehensive test coverage
- Easy to test and maintain

## Testing Mode

For rapid development and testing:

```python
# In main_2025_26.py
TESTING_MODE = True  # Only scrapes 2 divisions
```

**Benefits:**
- 4-6 minutes vs 60-90 minutes
- Same code paths as production
- Validates core functionality
- Faster iteration during development

## Recent Improvements (October 2025)

### 🎯 Refactoring & Testing

**Phase 1: Parser Extraction**
- ✅ Extracted 8 parsing functions to `parsers.py`
- ✅ Created 58 comprehensive unit tests
- ✅ Fixed pandas warnings (FutureWarning, SettingWithCopyWarning)
- ✅ Added TESTING_MODE for fast development

**Phase 2: Scraper Modularization**
- ✅ Created `scrapers/` package with 5 specialized scrapers
- ✅ Implemented `BaseScraper` with retry logic and session management
- ✅ Removed 400+ lines of duplicate code from main script
- ✅ Improved maintainability and testability

**Phase 3: Scraper Testing**
- ✅ Created 20 unit tests with mocked HTTP responses
- ✅ Tests run offline in 30 seconds
- ✅ Platform-agnostic type assertions
- ✅ Comprehensive edge case coverage

**Total Impact:**
- 78 tests (all passing)
- Reduced main script by 400+ lines
- Improved code organization
- Enhanced reliability
- Faster development cycle

## Dependencies

Key packages:
- `beautifulsoup4` - HTML parsing
- `pandas` - Data manipulation
- `requests` - HTTP client
- `streamlit` - Dashboard framework
- `pytest` - Testing framework

See `requirements.txt` for complete list.

## Data Files

### Current Season (2025-2026/)
```
├── schedules_df/          # Match schedules by division/week
├── results_df/            # Match results
├── summary_df/            # Team standings
├── teams_df/              # Team rosters
├── players_df/            # Player rosters
├── ranking_df/            # Player rankings
├── player_results/        # Individual match results
├── detailed_league_tables/ # Full league tables
├── home_away_data/        # Home/away analysis
├── team_win_percentage_breakdown/ # Win % by rubber position
└── combined_*.csv         # Aggregated data across divisions
```

### Historical Seasons
```
previous_seasons/
├── 2016-2017/
├── 2017-2018/
├── 2018-2019/
├── 2019-2020/
├── 2021-2022/
├── 2022-2023/
├── 2023-2024/
└── 2024-2025/
```

## Contributing

### Branch Strategy

- `main` - Stable production branch
- `feature/*` - Feature development branches

Current development branch: `feature/refactor-parsers-and-testing`

### Making Changes

1. Create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass (`pytest tests/ -v`)
4. Update documentation
5. Submit pull request

### Code Style

- Follow PEP 8 conventions
- Write descriptive function names
- Add docstrings to functions
- Keep functions focused (single responsibility)
- Use type hints where appropriate

## Error Handling

All scrapers implement robust error handling:
- **HTTP Errors**: Retry with exponential backoff
- **Timeouts**: 30-second timeout per request
- **Empty Pages**: Return empty DataFrames
- **Malformed Data**: Log warnings and skip bad data
- **NO DATA**: Handle gracefully (common for certain divisions)

## Logging

Configure logging level in scripts:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**Log Levels:**
- `INFO` - Successful operations, progress updates
- `WARNING` - Recoverable issues, missing data
- `ERROR` - Failed operations, critical issues

## Performance

**Full Scrape (28 divisions):**
- Duration: 60-90 minutes
- Requests: ~150 HTTP requests
- Data: ~500+ CSV files generated

**Testing Mode (2 divisions):**
- Duration: 4-6 minutes
- Requests: ~10 HTTP requests
- Data: Subset of CSV files

**Test Suite:**
- Duration: ~35 seconds
- Tests: 78 tests
- Coverage: Comprehensive (scrapers + parsers)

## Known Issues

1. **Website Typo**: HK Squash website uses "team_summery" (misspelled) - handled with fallback logic
2. **NO DATA Teams**: Some divisions have teams with no player data - gracefully skipped
3. **Inconsistent HTML**: Website HTML structure varies slightly - robust parsing handles this

## Future Improvements

- [ ] Add integration tests for full workflow
- [ ] Implement data validation layer
- [ ] Add API endpoint for programmatic access
- [ ] Create automated daily scraping schedule
- [ ] Add player ELO rating system
- [ ] Implement match prediction models
- [ ] Add email notifications for results
- [ ] Create mobile-responsive dashboard

## License

This project is for personal use. Data is sourced from the Hong Kong Squash website.

## Contact

**Repository**: https://github.com/bpal852/SquashApp  
**Branch**: feature/refactor-parsers-and-testing

## Acknowledgments

- Hong Kong Squash for providing match data
- Streamlit for the dashboard framework
- pytest for the excellent testing framework
