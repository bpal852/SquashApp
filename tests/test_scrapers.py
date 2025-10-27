"""
Unit tests for scraper modules.

Tests all scrapers with mocked HTTP responses to ensure parsing logic works correctly
without making actual network requests.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from bs4 import BeautifulSoup

from scrapers.teams import TeamsScraper
from scrapers.summary import TeamSummaryScraper
from scrapers.schedules import SchedulesAndResultsScraper
from scrapers.ranking import RankingScraper
from scrapers.players import PlayersScraper


# ============================================================================
# Test Fixtures - Reusable mock data
# ============================================================================

@pytest.fixture
def mock_session():
    """Create a mock session that can be configured per test"""
    session = Mock()
    return session


@pytest.fixture
def teams_html():
    """Sample HTML for teams page"""
    return b"""
    <html>
        <body>
            <div class="teams-content-list">
                <div>Kowloon Cricket Club 2</div>
                <div>Hong Kong Cricket Club</div>
                <div>John Smith</div>
                <div>john.smith@kcc.com</div>
            </div>
            <div class="teams-content-list">
                <div>Hong Kong Football Club 2A</div>
                <div>HKFC</div>
                <div>Jane Doe</div>
                <div>jane.doe@hkfc.com</div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def summary_html():
    """Sample HTML for team summary page"""
    return b"""
    <html>
        <body>
            <div class="clearfix teamSummary-content-list">
                <div class="col-xs-4">Kowloon Cricket Club 2</div>
                <div class="col-xs-2">2</div>
                <div class="col-xs-2">2</div>
                <div class="col-xs-2">0</div>
                <div class="col-xs-2">10</div>
            </div>
            <div class="clearfix teamSummary-content-list">
                <div class="col-xs-4">Hong Kong Football Club 2A</div>
                <div class="col-xs-2">2</div>
                <div class="col-xs-2">1</div>
                <div class="col-xs-2">1</div>
                <div class="col-xs-2">5</div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def schedules_html():
    """Sample HTML for schedules and results page"""
    return b"""
    <html>
        <body>
            <div class="clearfix results-schedules-title">Week 1 - 2025-09-09</div>
            <div class="results-schedules-content">
                <div class="results-schedules-list">
                    <div>Home Team</div>
                    <div>vs</div>
                    <div>Away Team</div>
                    <div>Venue</div>
                    <div>Time</div>
                    <div>Result</div>
                </div>
                <div class="results-schedules-list">
                    <div>Kowloon CC 2</div>
                    <div>vs</div>
                    <div>HKFC 2A</div>
                    <div>HKCC</div>
                    <div>19:00</div>
                    <div>5-0</div>
                </div>
                <div class="results-schedules-list">
                    <div>HKFC 2B</div>
                    <div>vs</div>
                    <div>Young Player 2</div>
                    <div>HKFC</div>
                    <div>20:00</div>
                    <div></div>
                </div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def ranking_html():
    """Sample HTML for ranking page"""
    return b"""
    <html>
        <body>
            <a href="/leagues/detail/id/D00473">Division 2</a>
            <div class="clearfix ranking-content-list">
                <div>1</div>
                <div>John Smith</div>
                <div>Kowloon CC 2</div>
                <div>4.5</div>
                <div>27</div>
                <div>6</div>
                <div>5</div>
                <div>1</div>
            </div>
            <div class="clearfix ranking-content-list">
                <div>2</div>
                <div>Jane Doe</div>
                <div>HKFC 2A</div>
                <div>4.0</div>
                <div>20</div>
                <div>5</div>
                <div>4</div>
                <div>1</div>
            </div>
            <div class="clearfix ranking-content-list">
                <div>3</div>
                <div>Bob Wilson</div>
                <div>HKFC 2A</div>
                <div>3.5</div>
                <div>14</div>
                <div>4</div>
                <div>2</div>
                <div>2</div>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def players_html():
    """Sample HTML for players page"""
    return b"""
    <html>
        <body>
            <div class="players-container">
                <div>team name:</div>
                <div>Kowloon Cricket Club 2</div>
                <div class="players-content-list">
                    <div class="col-xs-2">1</div>
                    <div class="col-xs-4">John Smith</div>
                    <div class="col-xs-2">12345</div>
                    <div class="col-xs-2">150</div>
                    <div class="col-xs-2">4.5</div>
                </div>
                <div class="players-content-list">
                    <div class="col-xs-2">2</div>
                    <div class="col-xs-4">Jane Doe</div>
                    <div class="col-xs-2">12346</div>
                    <div class="col-xs-2">160</div>
                    <div class="col-xs-2">4.0</div>
                </div>
            </div>
            <div class="players-container">
                <div>team name:</div>
                <div>HKFC 2A</div>
                <div class="players-content-list">
                    <div class="col-xs-2">1</div>
                    <div class="col-xs-4">Bob Wilson</div>
                    <div class="col-xs-2">12347</div>
                    <div class="col-xs-2">140</div>
                    <div class="col-xs-2">3.5</div>
                </div>
            </div>
        </body>
    </html>
    """


# ============================================================================
# TeamsScraper Tests
# ============================================================================

class TestTeamsScraper:
    """Test cases for TeamsScraper"""
    
    def test_successful_scrape(self, mock_session, teams_html):
        """Test successful scraping with valid HTML"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = teams_html
        mock_session.get.return_value = mock_response
        
        # Execute
        scraper = TeamsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        # Verify
        assert len(df) == 2
        assert list(df.columns) == ["Team Name", "Home", "Convenor", "Email"]
        assert df.iloc[0]['Team Name'] == 'Kowloon Cricket Club 2'
        assert df.iloc[0]['Home'] == 'Hong Kong Cricket Club'
        assert df.iloc[0]['Convenor'] == 'John Smith'
        assert df.iloc[0]['Email'] == 'john.smith@kcc.com'
        assert df.iloc[1]['Team Name'] == 'Hong Kong Football Club 2A'
    
    def test_empty_page(self, mock_session):
        """Test handling of page with no team data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html><body>NO DATA</body></html>"
        mock_session.get.return_value = mock_response
        
        scraper = TeamsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert df.empty
    
    def test_http_error_404(self, mock_session):
        """Test handling of 404 error"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.content = b"Not Found"
        mock_session.get.return_value = mock_response
        
        scraper = TeamsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert df.empty
    
    def test_malformed_html(self, mock_session):
        """Test handling of malformed HTML"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"""
        <html><body>
            <div class="teams-content-list">
                <div>Team Only</div>
            </div>
        </body></html>
        """
        mock_session.get.return_value = mock_response
        
        scraper = TeamsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        # Should handle gracefully - may be empty or partial
        assert isinstance(df, pd.DataFrame)


# ============================================================================
# TeamSummaryScraper Tests
# ============================================================================

class TestTeamSummaryScraper:
    """Test cases for TeamSummaryScraper"""
    
    def test_successful_scrape_summery_spelling(self, mock_session, summary_html):
        """Test successful scraping with 'team_summery' spelling"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = summary_html
        mock_session.get.return_value = mock_response
        
        scraper = TeamSummaryScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 2
        assert list(df.columns) == ["Team", "Played", "Won", "Lost", "Points"]
        assert df.iloc[0]['Team'] == 'Kowloon Cricket Club 2'
        assert df.iloc[0]['Played'] == 2
        assert df.iloc[0]['Won'] == 2
        assert df.iloc[0]['Lost'] == 0
        assert df.iloc[0]['Points'] == 10
        
        # Check data types (use kind='i' to allow int32 or int64)
        assert df['Played'].dtype.kind == 'i'  # integer type
        assert df['Won'].dtype.kind == 'i'
        assert df['Lost'].dtype.kind == 'i'
        assert df['Points'].dtype.kind == 'i'
    
    def test_fallback_to_summary_spelling(self, mock_session, summary_html):
        """Test fallback to 'team_summary' spelling if first fails"""
        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 404
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.content = summary_html
        
        mock_session.get.side_effect = [mock_response_fail, mock_response_success]
        
        scraper = TeamSummaryScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 2
        assert df.iloc[0]['Team'] == 'Kowloon Cricket Club 2'
    
    def test_both_spellings_fail(self, mock_session):
        """Test that SystemExit is raised when both spellings fail"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        
        scraper = TeamSummaryScraper(session=mock_session)
        
        with pytest.raises(SystemExit):
            scraper.scrape("D00473", "2025-2026")
    
    def test_header_row_filtering(self, mock_session):
        """Test that header rows are filtered out"""
        html_with_header = b"""
        <html><body>
            <div class="clearfix teamSummary-content-list">
                <div>Team</div>
                <div>Played</div>
                <div>Won</div>
                <div>Lost</div>
                <div>Points</div>
            </div>
            <div class="clearfix teamSummary-content-list">
                <div class="col-xs-4">Real Team</div>
                <div class="col-xs-2">1</div>
                <div class="col-xs-2">1</div>
                <div class="col-xs-2">0</div>
                <div class="col-xs-2">5</div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html_with_header
        mock_session.get.return_value = mock_response
        
        scraper = TeamSummaryScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 1
        assert df.iloc[0]['Team'] == 'Real Team'


# ============================================================================
# SchedulesAndResultsScraper Tests
# ============================================================================

class TestSchedulesAndResultsScraper:
    """Test cases for SchedulesAndResultsScraper"""
    
    def test_successful_scrape(self, mock_session, schedules_html):
        """Test successful scraping with valid HTML"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = schedules_html
        mock_session.get.return_value = mock_response
        
        scraper = SchedulesAndResultsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 2
        assert list(df.columns) == ['Home Team', 'vs', 'Away Team', 'Venue', 'Time', 'Result', 'Match Week', 'Date']
        
        # Check first row (completed match)
        assert df.iloc[0]['Home Team'] == 'Kowloon CC 2'
        assert df.iloc[0]['Away Team'] == 'HKFC 2A'
        assert df.iloc[0]['Result'] == '5-0'
        assert df.iloc[0]['Match Week'] == 1
        assert df.iloc[0]['Date'] == '2025-09-09'
        
        # Check second row (upcoming match)
        assert df.iloc[1]['Home Team'] == 'HKFC 2B'
        assert df.iloc[1]['Result'] == ''  # No result yet
        assert df.iloc[1]['Match Week'] == 1
    
    def test_match_week_extraction(self, mock_session):
        """Test extraction of match week number"""
        html = b"""
        <html><body>
            <div class="clearfix results-schedules-title">Week 5 - 2025-10-15</div>
            <div class="results-schedules-content">
                <div class="results-schedules-list">
                    <div>Home</div><div>vs</div><div>Away</div><div>Venue</div><div>Time</div><div>Result</div>
                </div>
                <div class="results-schedules-list">
                    <div>Team A</div><div>vs</div><div>Team B</div><div>V1</div><div>19:00</div><div>3-2</div>
                </div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = SchedulesAndResultsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert df.iloc[0]['Match Week'] == 5
        assert df.iloc[0]['Date'] == '2025-10-15'
    
    def test_missing_result_column(self, mock_session):
        """Test handling when result column is missing"""
        html = b"""
        <html><body>
            <div class="clearfix results-schedules-title">Week 1 - 2025-09-09</div>
            <div class="results-schedules-content">
                <div class="results-schedules-list">
                    <div>Home</div><div>vs</div><div>Away</div><div>Venue</div><div>Time</div>
                </div>
                <div class="results-schedules-list">
                    <div>Team A</div><div>vs</div><div>Team B</div><div>V1</div><div>19:00</div>
                </div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = SchedulesAndResultsScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 1
        assert df.iloc[0]['Result'] == ''  # Should add empty result


# ============================================================================
# RankingScraper Tests
# ============================================================================

class TestRankingScraper:
    """Test cases for RankingScraper"""
    
    def test_successful_scrape(self, mock_session, ranking_html):
        """Test successful scraping with valid HTML"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = ranking_html
        mock_session.get.return_value = mock_response
        
        scraper = RankingScraper(session=mock_session)
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scraper.scrape("D00473", "2025-2026")
        
        # Check main ranking DataFrame
        assert len(ranking_df) == 3
        assert 'Win Percentage' in ranking_df.columns
        assert ranking_df.iloc[0]['Name of Player'] == 'John Smith'
        assert ranking_df.iloc[0]['Games Played'] == 6
        assert ranking_df.iloc[0]['Won'] == 5
        assert ranking_df.iloc[0]['Lost'] == 1
        
        # Check win percentage calculation
        assert ranking_df.iloc[0]['Win Percentage'] == pytest.approx(5/6)
        
        # Check filtered DataFrame (>= 5 games)
        assert len(ranking_df_filtered) == 2  # Bob has only 4 games
        
        # Check Division column
        assert all(ranking_df['Division'] == '2')
    
    def test_filtering_less_than_5_games(self, mock_session):
        """Test that players with < 5 games are filtered"""
        html = b"""
        <html><body>
            <a href="/leagues/detail/id/D00473">Division 2</a>
            <div class="clearfix ranking-content-list">
                <div>1</div><div>Player 1</div><div>Team A</div>
                <div>4.5</div><div>27</div><div>6</div><div>5</div><div>1</div>
            </div>
            <div class="clearfix ranking-content-list">
                <div>2</div><div>Player 2</div><div>Team A</div>
                <div>3.0</div><div>6</div><div>2</div><div>1</div><div>1</div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = RankingScraper(session=mock_session)
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scraper.scrape("D00473", "2025-2026")
        
        assert len(ranking_df) == 2
        assert len(ranking_df_filtered) == 1  # Only Player 1 has >= 5 games
        assert ranking_df_filtered.iloc[0]['Name of Player'] == 'Player 1'
    
    def test_unbeaten_players(self, mock_session):
        """Test identification of unbeaten players"""
        html = b"""
        <html><body>
            <a href="/leagues/detail/id/D00473">Division 2</a>
            <div class="clearfix ranking-content-list">
                <div>1</div><div>Unbeaten Player</div><div>Team A</div>
                <div>5.0</div><div>30</div><div>6</div><div>6</div><div>0</div>
            </div>
            <div class="clearfix ranking-content-list">
                <div>2</div><div>Regular Player</div><div>Team A</div>
                <div>4.0</div><div>20</div><div>5</div><div>3</div><div>2</div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = RankingScraper(session=mock_session)
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scraper.scrape("D00473", "2025-2026")
        
        assert len(unbeaten_list) == 1
        assert 'Unbeaten Player (Team A)' in unbeaten_list
    
    def test_no_data_handling(self, mock_session):
        """Test handling when NO DATA is present"""
        html = b"""
        <html><body>
            <div class="clearfix ranking-content-list">
                <div>NO DATA</div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = RankingScraper(session=mock_session)
        ranking_df, summarized_df, unbeaten_list, ranking_df_filtered = scraper.scrape("D00473", "2025-2026")
        
        assert ranking_df is None
        assert summarized_df is None
        assert unbeaten_list == []
        assert ranking_df_filtered is None


# ============================================================================
# PlayersScraper Tests
# ============================================================================

class TestPlayersScraper:
    """Test cases for PlayersScraper"""
    
    def test_successful_scrape(self, mock_session, players_html):
        """Test successful scraping with valid HTML"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = players_html
        mock_session.get.return_value = mock_response
        
        scraper = PlayersScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 3
        assert list(df.columns) == ['Order', 'Player', 'HKS No.', 'Ranking', 'Points', 'Team']
        
        # Check first player
        assert df.iloc[0]['Player'] == 'John Smith'
        assert df.iloc[0]['Order'] == 1
        assert df.iloc[0]['HKS No.'] == 12345
        assert df.iloc[0]['Ranking'] == 150
        assert df.iloc[0]['Points'] == 4.5
        assert df.iloc[0]['Team'] == 'Kowloon Cricket Club 2'
        
        # Check player from second team
        assert df.iloc[2]['Player'] == 'Bob Wilson'
        assert df.iloc[2]['Team'] == 'HKFC 2A'
    
    def test_no_data_team_handling(self, mock_session):
        """Test handling of teams with NO DATA"""
        html = b"""
        <html><body>
            <div class="players-container">
                <div>team name:</div>
                <div>Team A</div>
                <div class="players-content-list">
                    <div class="col-xs-2">1</div>
                    <div class="col-xs-4">Player 1</div>
                    <div class="col-xs-2">12345</div>
                    <div class="col-xs-2">150</div>
                    <div class="col-xs-2">4.5</div>
                </div>
            </div>
            <div class="players-container">
                <div>team name:</div>
                <div>Team B</div>
                <div>NO DATA</div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = PlayersScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 1  # Only Team A should be included
        assert df.iloc[0]['Team'] == 'Team A'
    
    def test_malformed_player_rows(self, mock_session):
        """Test handling of malformed player rows"""
        html = b"""
        <html><body>
            <div class="players-container">
                <div>team name:</div>
                <div>Team A</div>
                <div class="players-content-list">
                    <div class="col-xs-2">1</div>
                    <div class="col-xs-4">Valid Player</div>
                    <div class="col-xs-2">12345</div>
                    <div class="col-xs-2">150</div>
                    <div class="col-xs-2">4.5</div>
                </div>
                <div class="players-content-list">
                    <div class="col-xs-2">Invalid</div>
                    <div class="col-xs-4">Header Row</div>
                </div>
            </div>
        </body></html>
        """
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = PlayersScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        assert len(df) == 1  # Only valid player should be included
        assert df.iloc[0]['Player'] == 'Valid Player'
    
    def test_numeric_conversions(self, mock_session, players_html):
        """Test that numeric fields are properly converted"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = players_html
        mock_session.get.return_value = mock_response
        
        scraper = PlayersScraper(session=mock_session)
        df = scraper.scrape("D00473", "2025-2026")
        
        # Check data types (use kind to allow int32/int64, float32/float64)
        assert df['Order'].dtype.kind == 'i'  # integer type
        assert df['HKS No.'].dtype.kind == 'i'  # integer type
        assert df['Ranking'].dtype.kind == 'i'  # integer type
        assert df['Points'].dtype.kind == 'f'  # float type
    
    def test_no_teams_found(self, mock_session):
        """Test handling when no valid teams are found"""
        html = b"<html><body>NO DATA</body></html>"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = html
        mock_session.get.return_value = mock_response
        
        scraper = PlayersScraper(session=mock_session)
        
        with pytest.raises(ValueError, match="No valid player data found"):
            scraper.scrape("D00473", "2025-2026")


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
