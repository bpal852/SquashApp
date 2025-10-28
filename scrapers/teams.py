"""
Teams page scraper for HK Squash website.

Scrapes team information including team names, home venues, convenors, and emails.
"""

import pandas as pd
from typing import Optional
from .base import BaseScraper


def scrape_teams_page(league_id: str, year: str, session=None) -> pd.DataFrame:
    """
    Scrape the Teams page from the HK Squash website.
    
    Args:
        league_id: The league ID (e.g., "D00473")
        year: The year string (e.g., "2025-2026")
        session: Optional requests.Session to use
    
    Returns:
        DataFrame with columns: Team Name, Home, Convenor, Email
        Returns empty DataFrame if scraping fails
    """
    scraper = TeamsScraper(session)
    return scraper.scrape(league_id, year)


class TeamsScraper(BaseScraper):
    """Scraper for the teams page."""
    
    def scrape(self, league_id: str, year: str) -> pd.DataFrame:
        """
        Scrape teams data for a specific league and year.
        
        Args:
            league_id: The league ID (e.g., "D00473")
            year: The year string (e.g., "2025-2026")
        
        Returns:
            DataFrame with team information or empty DataFrame on failure
        """
        # Extract division name from league_id for logging
        division = league_id.replace("D00", "")
        
        self.log_scrape_start("Teams", division, league_id, year)
        
        try:
            # Build URL and fetch page
            teams_url = self.build_url("teams", league_id, year)
            self.logger.debug(f"Constructed teams URL: {teams_url}")
            
            soup = self.fetch_page(teams_url)
            if soup is None:
                return pd.DataFrame()
            
            # Find the team data
            team_rows = soup.find_all("div", class_="teams-content-list")
            self.logger.debug(f"Found {len(team_rows)} team rows")
            
            # Check if any team data was found
            if not team_rows:
                self.logger.warning("No team data was found on the teams page.")
                return pd.DataFrame()
            
            # Extract data from rows
            team_data_rows = []
            for idx, row in enumerate(team_rows):
                columns = row.find_all("div", recursive=False)
                row_data = [col.text.strip() for col in columns if col.text.strip()]
                
                if row_data:
                    team_data_rows.append(row_data)
                    self.logger.debug(f"Extracted data from row {idx}: {row_data}")
                else:
                    self.logger.debug(f"No data found in row {idx}, skipping")
            
            # Check if any data was extracted
            if not team_data_rows:
                self.logger.warning("No data rows were extracted from the teams page.")
                return pd.DataFrame()
            
            # Define the expected column names
            expected_columns = ["Team Name", "Home", "Convenor", "Email"]
            
            # Create DataFrame from list of lists
            teams_df = pd.DataFrame(team_data_rows, columns=expected_columns)
            
            self.log_scrape_success("Teams", division, len(teams_df))
            
            return teams_df
            
        except Exception as e:
            self.log_scrape_error("Teams", division, e)
            return pd.DataFrame()
