# Tests

This directory contains comprehensive unit tests for the SquashApp project.

## Test Files

### `test_parsers.py` (58 tests)

Unit tests for all parsing functions in `parsers.py`.

**Coverage:**
- `parse_result()` - 12 tests (normal scores, WO, CR, ties, invalid input)
- `count_games_won()` - 6 tests (various scores, WO/CR handling)
- `normalize_rubber()` - 8 tests (formatting, WO/CR, whitespace)
- `determine_winner()` - 8 tests (home/away wins, ties, edge cases)
- `format_rubbers()` - 6 tests (list/string input, WO/CR, formatting)
- `parse_home_away_scores()` - 6 tests (normal scores, WO/CR, invalid)
- `extract_match_details()` - 6 tests (complete/incomplete data, missing fields)
- `extract_match_result()` - 6 tests (normal results, WO/CR, empty results)

**Running:**
```powershell
# Run all parser tests
pytest tests/test_parsers.py -v

# Run specific test
pytest tests/test_parsers.py::test_parse_result_normal_score -v
```

---

### `test_scrapers.py` (20 tests)

Unit tests for all scraper modules in `scrapers/` package using mocked HTTP responses.

**Coverage:**

#### TestTeamsScraper (4 tests)
- `test_successful_scrape` - Verifies DataFrame structure and data
- `test_empty_page` - Handles pages with no teams
- `test_http_error_404` - Proper error handling for 404s
- `test_malformed_html` - Handles unexpected HTML structure

#### TestTeamSummaryScraper (4 tests)
- `test_successful_scrape_summery_spelling` - Tests "team_summery" (misspelled) URL
- `test_fallback_to_summary_spelling` - Verifies fallback to correct spelling
- `test_both_spellings_fail` - Ensures SystemExit when both fail
- `test_header_row_filtering` - Removes "Team" header rows

#### TestSchedulesAndResultsScraper (3 tests)
- `test_successful_scrape` - Verifies complete data extraction
- `test_match_week_extraction` - Tests week and date parsing
- `test_missing_result_column` - Handles fixtures without results

#### TestRankingScraper (4 tests)
- `test_successful_scrape` - Verifies all return values
- `test_filtering_less_than_5_games` - Ensures <5 games filtered out
- `test_unbeaten_players` - Validates unbeaten player identification
- `test_no_data_handling` - Handles "NO DATA" pages gracefully

#### TestPlayersScraper (5 tests)
- `test_successful_scrape` - Verifies DataFrame structure
- `test_no_data_team_handling` - Skips teams with "NO DATA"
- `test_malformed_player_rows` - Skips invalid rows
- `test_numeric_conversions` - Validates data type conversions
- `test_no_teams_found` - Raises ValueError when no teams exist

**Running:**
```powershell
# Run all scraper tests
pytest tests/test_scrapers.py -v

# Run specific test class
pytest tests/test_scrapers.py::TestTeamsScraper -v

# Run specific test
pytest tests/test_scrapers.py::TestRankingScraper::test_unbeaten_players -v
```

---

## Testing Strategy

### 1. Parser Tests (Pure Functions)

Parser tests use **direct function calls** with various input scenarios:
- Normal cases
- Edge cases
- Error conditions
- Type validation

**Example:**
```python
def test_parse_result_normal_score():
    """Test parsing a normal score like '3-2'"""
    assert parse_result("3-2") == (3, 2)
```

### 2. Scraper Tests (Mocked HTTP)

Scraper tests use **`unittest.mock`** to mock HTTP responses, eliminating network dependencies:

**Benefits:**
- ✅ **Fast** - No network calls (30s for all 20 tests)
- ✅ **Reliable** - No external dependencies
- ✅ **Offline** - Run without internet connection
- ✅ **Controlled** - Test exact scenarios including edge cases

**Example:**
```python
def test_successful_scrape(self, mock_session, teams_html):
    """Test successful scraping with mocked HTTP response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = teams_html
    mock_session.get.return_value = mock_response
    
    scraper = TeamsScraper(session=mock_session)
    df = scraper.scrape("D00473", "2025-2026")
    
    assert len(df) == 2
    assert list(df.columns) == ["Team Name", "Home", "Convenor", "Email"]
```

### 3. Fixtures

Pytest fixtures provide reusable test data:

**`mock_session`** - Mocked requests.Session object
```python
@pytest.fixture
def mock_session():
    return Mock(spec=requests.Session)
```

**HTML Fixtures** - Realistic HTML responses for each scraper type:
- `teams_html` - Team roster page
- `summary_html` - Match summary page
- `schedules_html` - Fixtures/results page
- `ranking_html` - Player rankings page
- `players_html` - Player details page

---

## Running Tests

### All Tests
```powershell
# Run all tests in tests directory
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=scrapers --cov=parsers --cov-report=html
```

### Specific Test Files
```powershell
# Parser tests only
pytest tests/test_parsers.py -v

# Scraper tests only
pytest tests/test_scrapers.py -v
```

### Specific Test Classes
```powershell
# Test one scraper class
pytest tests/test_scrapers.py::TestTeamsScraper -v
```

### Specific Tests
```powershell
# Test one specific function
pytest tests/test_parsers.py::test_parse_result_normal_score -v
pytest tests/test_scrapers.py::TestRankingScraper::test_unbeaten_players -v
```

### Verbose Output
```powershell
# Show detailed output for each test
pytest tests/ -v

# Show print statements
pytest tests/ -v -s

# Show test durations
pytest tests/ -v --durations=10
```

---

## Test Results

### Current Status
- **Parser Tests**: ✅ 58/58 passing
- **Scraper Tests**: ✅ 20/20 passing
- **Total**: ✅ 78/78 passing
- **Runtime**: ~35 seconds total

### Platform Compatibility

Tests are platform-agnostic and work on:
- ✅ Windows (tested)
- ✅ Linux (compatible)
- ✅ macOS (compatible)

Data type assertions use `.dtype.kind` to handle platform differences:
- Windows: `int32`, `float32`
- Linux/Mac: `int64`, `float64`

---

## Writing New Tests

### For Parser Functions

1. Import the function to test
2. Write test cases for normal, edge, and error scenarios
3. Use descriptive test names
4. Add docstrings explaining what's being tested

```python
def test_my_parser_function_normal_case():
    """Test my_parser_function with typical input"""
    result = my_parser_function("input")
    assert result == expected_output
```

### For Scraper Classes

1. Create pytest fixtures for HTML responses
2. Mock the session and HTTP responses
3. Test successful cases AND error cases
4. Validate DataFrame structure and data types

```python
@pytest.fixture
def my_html():
    return b"<html>...</html>"

def test_my_scraper(self, mock_session, my_html):
    """Test MyScraper with mocked HTTP response"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = my_html
    mock_session.get.return_value = mock_response
    
    scraper = MyScraper(session=mock_session)
    df = scraper.scrape("D00473", "2025-2026")
    
    assert len(df) > 0
    assert 'expected_column' in df.columns
```

---

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ -v --cov=scrapers --cov=parsers
```

---

## Test Coverage Goals

Current coverage is excellent:
- ✅ All parser functions have comprehensive tests
- ✅ All scraper classes have comprehensive tests
- ✅ Error conditions are tested
- ✅ Edge cases are covered

Future improvements:
- [ ] Add integration tests for full workflow
- [ ] Add performance benchmarks
- [ ] Add tests for app.py and main scripts

---

## Troubleshooting

### Tests Fail with Import Errors
```powershell
# Ensure you're in the project root
cd C:\Users\bpali\PycharmProjects\SquashApp

# Install in development mode
pip install -e .
```

### Tests Fail with "pytest not found"
```powershell
# Install pytest in virtual environment
venv\Scripts\python.exe -m pip install pytest
```

### Mock Tests Not Working
```powershell
# Ensure unittest.mock is available (Python 3.3+)
python --version  # Should be 3.11.4

# unittest.mock is in standard library
```

### Platform-Specific Type Errors
Use `.dtype.kind` instead of exact type names:
```python
# ❌ Platform-specific
assert df['column'].dtype == 'int64'

# ✅ Platform-agnostic
assert df['column'].dtype.kind == 'i'  # any integer type
```

---

## Best Practices

1. **Run tests before committing** - Ensure all tests pass
2. **Write tests for new features** - Maintain coverage
3. **Use descriptive names** - Make test purpose clear
4. **Test edge cases** - Don't just test happy path
5. **Keep tests fast** - Use mocks for external dependencies
6. **Document complex tests** - Add docstrings explaining logic

---

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [BeautifulSoup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
