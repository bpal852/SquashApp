"""
Ranking page scraper for HK Squash website.

Scrapes player ranking information including position, points, games played, wins, and losses.
Also generates summary statistics and filtered data.
"""

import pandas as pd
from typing import Optional, Tuple
from .base import BaseScraper


def scrape_ranking_page(league_id: str, year: str, session=None) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], list, Optional[pd.DataFrame]
]:
    """
    Scrape the Ranking page from the HK Squash website.
    
    Args:
        league_id: The league ID (e.g., "D00473")
        year: The year string (e.g., "2025-2026")
        session: Optional requests.Session to use
    
    Returns:
        Tuple of (ranking_df, summarized_df, unbeaten_list, ranking_df_filtered):
        - ranking_df: Full ranking DataFrame or None
        - summarized_df: Team summary DataFrame or None
        - unbeaten_list: List of unbeaten players
        - ranking_df_filtered: Filtered ranking DataFrame (>= 5 games) or None
    
    Raises:
        Exception: If page retrieval fails
    """
    scraper = RankingScraper(session)
    return scraper.scrape(league_id, year)


class RankingScraper(BaseScraper):
    """Scraper for the ranking page."""
    
    MIN_GAMES_FOR_FILTER = 5
    
    def scrape(self, league_id: str, year: str) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], list, Optional[pd.DataFrame]
    ]:
        """
        Scrape ranking data for a specific league and year.
        
        Args:
            league_id: The league ID
            year: The year string
        
        Returns:
            Tuple of DataFrames and lists with ranking data
        """
        division = league_id.replace("D00", "")
        
        self.log_scrape_start("Ranking", division, league_id, year)
        
        # Build URL and fetch page
        ranking_url = self.build_url("ranking", league_id, year)
        self.logger.debug(f"Constructed ranking URL: {ranking_url}")
        
        soup = self.fetch_page(ranking_url)
        if soup is None:
            raise Exception(f"Failed to retrieve ranking page for {league_id}")
        
        # Extract ranking data
        df = self._extract_ranking_data(soup)
        if df is None:
            return None, None, [], None
        
        # Add division column
        df = self._add_division_column(df, soup)
        
        # Convert to numeric types
        df = self._convert_numeric_columns(df)
        
        # Calculate win percentage
        df["Win Percentage"] = df.apply(
            lambda row: row["Won"] / row["Games Played"] if row["Games Played"] > 0 else 0,
            axis=1
        )
        
        # Filter for players with enough games
        ranking_df_filtered = df[df["Games Played"] >= self.MIN_GAMES_FOR_FILTER]
        self.logger.info(
            f"Filtered ranking DataFrame to {len(ranking_df_filtered)} rows "
            f"with {self.MIN_GAMES_FOR_FILTER} or more games played"
        )
        
        # Generate summaries
        if ranking_df_filtered.empty:
            self.logger.warning("No players have played enough games to qualify for the table.")
            summarized_df = None
            unbeaten_list = []
        else:
            summarized_df = self._create_summary_dataframe(df, ranking_df_filtered)
            unbeaten_list = self._get_unbeaten_players(ranking_df_filtered)
        
        return df, summarized_df, unbeaten_list, ranking_df_filtered
    
    def _extract_ranking_data(self, soup) -> Optional[pd.DataFrame]:
        """Extract ranking data from parsed HTML."""
        ranking_rows = soup.find_all("div", class_="clearfix ranking-content-list")
        self.logger.debug(f"Found {len(ranking_rows)} ranking rows")
        
        ranking_data_rows = []
        
        for idx, row in enumerate(ranking_rows):
            columns = row.find_all("div", recursive=False)
            row_data = [col.text.strip() for col in columns]
            
            # Exclude rows that contain "NO DATA" or are empty
            if "NO DATA" in row_data or not row_data or len(row_data) < 8:
                self.logger.debug(f"Skipping row {idx} due to 'NO DATA' or insufficient data")
                continue
            
            ranking_data_rows.append(row_data)
            self.logger.debug(f"Extracted data from row {idx}: {row_data}")
        
        if not ranking_data_rows:
            self.logger.warning("No data rows were extracted from the ranking page.")
            return None
        
        df = pd.DataFrame(
            ranking_data_rows,
            columns=['Position', 'Name of Player', 'Team', 'Average Points',
                    'Total Game Points', 'Games Played', 'Won', 'Lost']
        )
        self.logger.info(f"Successfully created ranking DataFrame with {len(df)} rows")
        
        return df
    
    def _add_division_column(self, df: pd.DataFrame, soup) -> pd.DataFrame:
        """Add division column by extracting from page."""
        try:
            full_division_name = soup.find('a', href=lambda href: href and "leagues/detail/id" in href).text.strip()
            division_number = full_division_name.split("Division ")[-1]
            df['Division'] = division_number
            self.logger.debug(f"Extracted division number: {division_number}")
        except Exception as e:
            self.logger.warning(f"Error extracting division number: {e}")
            df['Division'] = ''
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns to appropriate types."""
        numeric_columns = ['Average Points', 'Total Game Points', 'Games Played', 'Won', 'Lost']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle NaN values
        df['Average Points'] = df['Average Points'].fillna(0.0).astype(float)
        df['Total Game Points'] = df['Total Game Points'].fillna(0).astype(int)
        df['Games Played'] = df['Games Played'].fillna(0).astype(int)
        df['Won'] = df['Won'].fillna(0).astype(int)
        df['Lost'] = df['Lost'].fillna(0).astype(int)
        
        self.logger.debug("Converted numeric columns to appropriate data types")
        
        return df
    
    def _create_summary_dataframe(self, df: pd.DataFrame, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Create summary DataFrame with team statistics."""
        teams = df['Team'].unique()
        summary_data = {
            'Team': [],
            'Most Games': [],
            'Most Wins': [],
            'Highest Win Percentage': []
        }
        
        for team in teams:
            summary_data['Team'].append(team)
            summary_data['Most Games'].append(self._find_max_players(filtered_df, team, 'Games Played'))
            summary_data['Most Wins'].append(self._find_max_players(filtered_df, team, 'Won'))
            summary_data['Highest Win Percentage'].append(self._find_max_win_percentage(filtered_df, team))
        
        summarized_df = pd.DataFrame(summary_data).sort_values("Team")
        self.logger.info(f"Created summarized DataFrame with {len(summarized_df)} teams")
        
        return summarized_df
    
    def _find_max_players(self, df: pd.DataFrame, team: str, column: str) -> str:
        """Find players with max value in a column, handling ties."""
        max_value = df[df['Team'] == team][column].max()
        players = df[(df['Team'] == team) & (df[column] == max_value)]['Name of Player']
        return ", ".join(players) + f" ({max_value})"
    
    def _find_max_win_percentage(self, df: pd.DataFrame, team: str) -> str:
        """Find players with max win percentage, handling ties."""
        max_value = df[df['Team'] == team]['Win Percentage'].max()
        players = df[(df['Team'] == team) & (df['Win Percentage'] == max_value)]['Name of Player']
        return ", ".join(players) + f" ({max_value * 100:.1f}%)"
    
    def _get_unbeaten_players(self, filtered_df: pd.DataFrame) -> list:
        """Get list of unbeaten players."""
        unbeaten_list = filtered_df[
            filtered_df["Lost"] == 0
        ].apply(lambda row: f"{row['Name of Player']} ({row['Team']})", axis=1).tolist()
        
        self.logger.info(f"Found {len(unbeaten_list)} unbeaten players")
        
        return unbeaten_list
