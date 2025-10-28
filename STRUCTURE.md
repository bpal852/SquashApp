# SquashApp Directory Structure

This document describes the organization of the SquashApp codebase.

## 📁 Root Directory Layout

```
SquashApp/
├── main_2025_26.py         # Main entry point for 2025-2026 season scraper
├── app_update.py           # Streamlit web application for data visualization
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
├── pytest.ini              # Pytest configuration
├── README.md               # Project documentation
├── STRUCTURE.md            # This file
└── .gitignore              # Git ignore rules
```

## 📦 Core Packages

### `config/` - Configuration Management
Environment-aware configuration system with validation.

```
config/
├── __init__.py             # Package exports
├── settings.py             # Configuration classes (BaseConfig, DevelopmentConfig, etc.)
├── README.md               # Configuration documentation
└── divisions/              # Division configuration files
    └── 2025-2026.json      # Current season divisions
```

**Usage:**
```python
from config import get_config
config = get_config('development')  # or 'testing', 'production'
```

### `scrapers/` - Web Scraping Modules
Modular scrapers for extracting data from HK Squash website.

```
scrapers/
├── __init__.py             # Package exports
├── base.py                 # BaseScraper with session management
├── teams.py                # TeamsScraper - team information
├── summary.py              # TeamSummaryScraper - match summaries
├── schedules.py            # SchedulesAndResultsScraper - fixtures/results
├── ranking.py              # RankingScraper - player rankings
├── players.py              # PlayersScraper - player rosters
└── README.md               # Scraper documentation
```

### `parsers/` - Data Parsing Functions
Pure functions for parsing scraped HTML/text data.

```
parsers/
├── __init__.py             # Package exports
├── results.py              # parse_result() - match results
├── scores.py               # split_overall_score() - score parsing
├── rubbers.py              # normalize_rubber() - rubber validation
└── summary.py              # parse_summary_row() - summary tables
```

### `validators/` - Data Validation
Comprehensive data validation for all scraped data.

```
validators/
├── __init__.py             # Package exports
├── base.py                 # BaseValidator, ValidationResult
├── teams.py                # TeamsValidator
├── summary.py              # SummaryValidator
├── schedules.py            # SchedulesValidator
├── ranking.py              # RankingValidator
├── players.py              # PlayersValidator
├── reports.py              # ValidationReportGenerator
└── README.md               # Validation documentation
```

### `utils/` - Utility Functions
Helper functions and utilities.

```
utils/
├── __init__.py
├── divisions_export.py     # save_divisions_json()
└── ...
```

## 🧪 Testing

### `tests/` - Unit Tests
Comprehensive test suite with 168+ tests.

```
tests/
├── __init__.py
├── test_config.py          # 40 tests - Configuration management
├── test_parsers.py         # 58 tests - Parser functions
├── test_scrapers.py        # 20 tests - Scraper modules
└── test_validators.py      # 50 tests - Data validation
```

**Run tests:**
```bash
pytest                      # Run all tests
pytest tests/test_config.py # Run specific test file
pytest -v                   # Verbose output
```

## 📊 Data Directories

### `data/` - Data Files
Organized storage for all data files.

```
data/
├── raw/                    # Raw/source data files
│   ├── hkcc_*.csv          # HKCC analysis data
│   └── updated_schedule_2024_2025.csv
├── processed/              # Processed/analyzed data
│   ├── big_summary_df.csv
│   ├── combined_player_results.csv
│   └── elo_results.csv
└── summer_league/          # Summer league specific data
    └── 2025 Admin - Summer League 2025 Round1Ratings.csv
```

### `outputs/` - Generated Outputs
HTML files and visualizations.

```
outputs/
├── hkcc_final_standings.html
├── matchups_table.html
└── hkcc_logo.png
```

### `logs/` - Log Files
Application log files.

```
logs/
├── app_debug.log
├── squash_app_debug.log
└── create_player_results_database_all_divisions.log
```

## 📂 Season Data

### `2025-2026/` - Current Season
Automatically generated data for current season.

```
2025-2026/
├── summary_df/             # League tables by week
├── teams_df/               # Team information
├── schedules_df/           # Fixtures and results
├── ranking_df/             # Player rankings
├── players_df/             # Player rosters
├── results_df/             # Match results
├── player_results/         # Individual player statistics
├── awaiting_results/       # Pending matches
├── remaining_fixtures/     # Upcoming matches
├── detailed_league_tables/ # Detailed standings
├── home_away_data/         # Home/away analysis
├── team_win_percentage_breakdown/  # Win % by rubber
├── simulated_tables/       # Projection simulations
├── simulated_fixtures/     # Simulated results
├── neutral_fixtures/       # Neutral venue matches
├── summarized_player_tables/  # Player summaries
├── unbeaten_players/       # Unbeaten player lists
├── played_every_game/      # Perfect attendance lists
├── logs/                   # Season-specific logs
└── combined_*.csv          # Combined datasets
```

### `previous_seasons/` - Historical Data
Archive of previous seasons (2016-2024).

```
previous_seasons/
├── 2016-2017/
├── 2017-2018/
├── ...
└── 2024-2025/
```

## 📜 Scripts

### `scripts/` - Utility Scripts
Helper scripts for data processing and analysis.

```
scripts/
├── check_missing_files.py
├── create_combined_results.py
├── create_player_results_database_all_divisions.py
├── projections.py
├── show_latest_awaiting_results.py
├── hkcc_filter_script.py
└── *.ipynb                 # Jupyter notebooks for analysis
```

## 🗄️ Archive

### `legacy/` - Archived Scripts
Old scripts from previous architecture (pre-refactoring).

```
legacy/
├── app.py                  # Old Streamlit app (superseded by app_update.py)
├── main.py                 # Old main script
├── main_update_2024_2025.py  # Old seasonal update
└── parsers.py              # Old monolithic parser (now in parsers/)
```

**Note:** These files are kept for reference but are no longer used. The functionality has been refactored into the modular packages. The current Streamlit app is `app_update.py` at the root.

## 🚀 Development Workflow

### Quick Start
```bash
# 1. Set up virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the scraper in development mode
$env:SQUASH_ENV = "development"  # Windows PowerShell
export SQUASH_ENV=development    # Mac/Linux
python main_2025_26.py

# 4. Run the Streamlit app
streamlit run app_update.py

# 5. Run tests
pytest -v
```

### Environment Variables
- `SQUASH_ENV` - Set environment (development/testing/production)
- `SQUASHAPP_ROOT` - Override repo root path (optional)

### Configuration Environments

| Environment | Divisions | Wait Time | Log Level | Use Case |
|-------------|-----------|-----------|-----------|----------|
| development | 3         | 5s        | DEBUG     | Local dev/testing |
| testing     | 2         | 1s        | WARNING   | Unit tests |
| production  | 36        | 30s       | INFO      | Full scraping runs |

## 📝 Notes

### Build Artifacts (Ignored by Git)
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `venv/` - Virtual environment
- `.idea/`, `.vscode/` - IDE settings
- `logs/`, `data/`, `outputs/` - Generated data

### Data Flow
1. **Scraping** → `scrapers/` fetch HTML from HK Squash website
2. **Parsing** → `parsers/` extract structured data
3. **Validation** → `validators/` verify data quality
4. **Storage** → Save to `2025-2026/` directories
5. **Analysis** → `scripts/` process and analyze data
6. **Output** → Generate reports in `outputs/`

### Key Files
- `main_2025_26.py` - Main orchestration script for scraping
- `app_update.py` - Streamlit web application for visualization
- `config/settings.py` - Centralized configuration
- `requirements.txt` - All Python dependencies
- `.gitignore` - What to exclude from Git

## 🔧 Maintenance

### Adding a New Season
1. Update `config/settings.py` → `SEASON_YEAR`
2. Create new division config JSON in `config/divisions/`
3. Run `main_2025_26.py` to generate season directories

### Adding a New Scraper
1. Create new file in `scrapers/`
2. Inherit from `BaseScraper`
3. Implement scraping logic
4. Add tests to `tests/test_scrapers.py`
5. Update `scrapers/__init__.py`

### Adding a New Validator
1. Create new file in `validators/`
2. Inherit from `BaseValidator`
3. Implement validation rules
4. Add tests to `tests/test_validators.py`
5. Update `validators/__init__.py`

## 📚 Documentation

- **README.md** - Main project documentation
- **config/README.md** - Configuration system guide
- **scrapers/README.md** - Scraper usage guide
- **validators/README.md** - Validation system guide
- **STRUCTURE.md** - This file

## 🧪 Testing Strategy

- **Unit Tests** - Test individual functions (parsers, validators)
- **Integration Tests** - Test module interactions (scrapers + parsers)
- **Mocked Tests** - Use `responses` library to mock HTTP requests
- **CI/CD** - GitHub Actions run tests on every push

### Test Coverage Goals
- Parsers: 100% (58 tests)
- Scrapers: 90%+ (20 tests)
- Validators: 100% (50 tests)
- Config: 100% (40 tests)

## 🎯 Project Goals

1. ✅ **Maintainability** - Modular, well-organized codebase
2. ✅ **Testability** - Comprehensive test coverage
3. ✅ **Reliability** - Data validation and error handling
4. ✅ **Scalability** - Easy to add new features
5. ✅ **Documentation** - Clear, comprehensive docs

---

**Last Updated:** October 28, 2025  
**Version:** 2.0 (Post-refactoring)
