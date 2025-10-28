"""
Players page scraper for HK Squash website.

Scrapes player information organized by team, including order, name, HKS No., ranking, and points.
"""

import pandas as pd
import time
from typing import Optional
from .base import BaseScraper


def scrape_players_page(league_id: str, year: str, session=None) -> Optional[pd.DataFrame]:
    """
    Scrape the Players page from the HK Squash website.
    
    Args:
        league_id: The league ID (e.g., "D00473")
        year: The year string (e.g., "2025-2026")
        session: Optional requests.Session to use
    
    Returns:
        DataFrame with columns: Order, Player, HKS No., Ranking, Points, Team
        Returns None if retrieval fails
    
    Raises:
        ValueError: If no valid player data found
        RuntimeError: If page retrieval fails
    """
    scraper = PlayersScraper(session)
    return scraper.scrape(league_id, year)


class PlayersScraper(BaseScraper):
    """Scraper for the players page."""
    
    SLEEP_BETWEEN_TEAMS = 5  # seconds to wait between processing teams
    
    def scrape(self, league_id: str, year: str) -> Optional[pd.DataFrame]:
        """
        Scrape player data for a specific league and year.
        
        Args:
            league_id: The league ID
            year: The year string
        
        Returns:
            DataFrame with player information or None
        """
        division = league_id.replace("D00", "")
        
        self.log_scrape_start("Players", division, league_id, year)
        
        # Build URL and fetch page
        players_url = self.build_url("players", league_id, year)
        self.logger.debug(f"Constructed players URL: {players_url}")
        
        soup = self.fetch_page(players_url)
        if soup is None:
            raise RuntimeError(f"Failed to retrieve players page for {league_id}")
        
        # Extract all team containers
        team_containers = soup.find_all("div", class_="players-container")
        self.logger.debug(f"Found {len(team_containers)} team containers")
        
        # Process each team
        team_dataframes = []
        
        for idx, team_container in enumerate(team_containers):
            team_df = self._process_team_container(team_container, idx)
            if team_df is not None:
                team_dataframes.append(team_df)
                time.sleep(self.SLEEP_BETWEEN_TEAMS)
        
        # Combine all team dataframes
        if not team_dataframes:
            raise ValueError("No valid player data found in any team block on the page.")
        
        combined_df = pd.concat(team_dataframes, ignore_index=True)
        self.logger.info(
            f"Concatenated all team dataframes into a single DataFrame with {len(combined_df)} rows"
        )
        
        return combined_df
    
    def _process_team_container(self, team_container, idx: int) -> Optional[pd.DataFrame]:
        """
        Process a single team container.
        
        Args:
            team_container: BeautifulSoup element containing team data
            idx: Index of the team (for logging)
        
        Returns:
            DataFrame with player data for this team or None
        """
        # Extract team name
        team_name = self._extract_team_name(team_container, idx)
        if team_name is None:
            return None
        
        # Check for NO DATA
        if team_container.get_text(strip=True).upper().find("NO DATA") != -1:
            self.logger.info(f"Team {idx} ('{team_name}') shows NO DATA — skipping team.")
            return None
        
        # Extract player rows
        player_rows = team_container.find_all("div", class_="players-content-list")
        self.logger.debug(f"Team {idx}: Found {len(player_rows)} player rows")
        
        players_data = self._extract_player_data(player_rows, idx)
        
        if not players_data:
            self.logger.warning(f"Team '{team_name}' produced no valid player rows; skipping team.")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(
            players_data, 
            columns=["Order", "Name of Players", "HKS No.", "Ranking", "Points"]
        )
        
        # Convert to correct data types
        df['Order'] = pd.to_numeric(df['Order'], errors='coerce').fillna(0).astype(int)
        df['HKS No.'] = pd.to_numeric(df['HKS No.'], errors='coerce').fillna(0).astype(int)
        df['Ranking'] = pd.to_numeric(df['Ranking'], errors='coerce').fillna(0).astype(int)
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce').fillna(0.0).astype(float)
        df['Team'] = team_name
        
        # Rename column
        df = df.rename(columns={"Name of Players": "Player"})
        
        self.logger.info(f"Team {idx + 1}: Created DataFrame with {len(df)} rows for team: {team_name}")
        
        return df
    
    def _extract_team_name(self, team_container, idx: int) -> Optional[str]:
        """Extract team name from team container."""
        try:
            team_name_div = team_container.find("div", string="team name:")
            team_name = team_name_div.find_next_sibling().get_text(strip=True)
            return team_name
        except Exception as e:
            self.logger.warning(f"Team {idx}: Error extracting team name: {e}")
            return None
    
    def _extract_player_data(self, player_rows, team_idx: int) -> list:
        """
        Extract player data from player rows.
        
        Args:
            player_rows: List of BeautifulSoup elements containing player data
            team_idx: Index of the team (for logging)
        
        Returns:
            List of player data rows
        """
        players_data = []
        
        for player_idx, player in enumerate(player_rows):
            # Collect fields
            order_rank_points = [
                div.get_text(strip=True) 
                for div in player.find_all("div", class_="col-xs-2")
            ]
            player_name = [
                div.get_text(strip=True) 
                for div in player.find_all("div", class_="col-xs-4")
            ]
            
            # Build row: [Order] + [Name of Players] + [HKS No., Ranking, Points]
            row = order_rank_points[:1] + player_name + order_rank_points[1:]
            
            # Keep only well-formed rows of length 5 with numeric order
            if len(row) == 5 and row[0].isdigit():
                players_data.append(row)
            else:
                # Skip headers/format noise
                self.logger.debug(
                    f"Team {team_idx}, Player {player_idx}: skipping malformed row: {row}"
                )
        
        return players_data
