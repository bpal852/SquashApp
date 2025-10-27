"""
Schedules and Results page scraper for HK Squash website.

Scrapes match schedules and results including teams, venues, times, and scores.
"""

import pandas as pd
from typing import Optional
from .base import BaseScraper


def scrape_schedules_and_results_page(league_id: str, year: str, session=None) -> pd.DataFrame:
    """
    Scrape the Schedules and Results page from the HK Squash website.
    
    Args:
        league_id: The league ID (e.g., "D00473")
        year: The year string (e.g., "2025-2026")
        session: Optional requests.Session to use
    
    Returns:
        DataFrame with columns: Home Team, vs, Away Team, Venue, Time, Result, Match Week, Date
        Returns empty DataFrame if scraping fails
    """
    scraper = SchedulesAndResultsScraper(session)
    return scraper.scrape(league_id, year)


class SchedulesAndResultsScraper(BaseScraper):
    """Scraper for the schedules and results page."""
    
    def scrape(self, league_id: str, year: str) -> pd.DataFrame:
        """
        Scrape schedules and results data for a specific league and year.
        
        Args:
            league_id: The league ID (e.g., "D00473")
            year: The year string (e.g., "2025-2026")
        
        Returns:
            DataFrame with schedule and results information or empty DataFrame on failure
        """
        # Extract division name from league_id for logging
        division = league_id.replace("D00", "")
        
        self.log_scrape_start("Schedules and Results", division, league_id, year)
        
        try:
            # Build URL and fetch page
            schedule_url = self.build_url("results_schedules", league_id, year)
            self.logger.debug(f"Constructed schedule URL: {schedule_url}")
            
            soup = self.fetch_page(schedule_url)
            if soup is None:
                return pd.DataFrame()
            
            # Initialize a list to hold all the data rows
            data_rows = []
            
            # Iterate over each section in the schedule
            sections = soup.find_all('div', class_='results-schedules-content')
            self.logger.debug(f"Found {len(sections)} schedule sections")
            
            for section_idx, section in enumerate(sections):
                # Extract the match week and date from the title
                match_week, date = self._extract_match_week_and_date(section, section_idx)
                
                # Find all schedule rows in the section
                schedule_rows = section.find_all('div', class_='results-schedules-list')
                self.logger.debug(f"Section {section_idx}: Found {len(schedule_rows)} schedule rows")
                
                # Skip the first row as it's the header
                for row_idx, row in enumerate(schedule_rows[1:], start=1):
                    row_data = self._extract_row_data(row, match_week, date, row_idx)
                    data_rows.append(row_data)
            
            # Create DataFrame from scraped data
            df = self._create_dataframe(data_rows)
            
            self.log_scrape_success("Schedules and Results", division, len(df))
            
            return df
            
        except Exception as e:
            self.log_scrape_error("Schedules and Results", division, e)
            return pd.DataFrame()
    
    def _extract_match_week_and_date(self, section, section_idx: int) -> tuple:
        """
        Extract match week and date from section title.
        
        Args:
            section: BeautifulSoup section element
            section_idx: Section index for logging
        
        Returns:
            Tuple of (match_week, date) or (None, None) if not found
        """
        title_div = section.find_previous_sibling('div', class_='clearfix results-schedules-title')
        
        if title_div:
            match_week_and_date = title_div.text.strip()
            try:
                match_week_str, date = match_week_and_date.split(' - ')
                # Extract just the number from the match week string
                match_week = ''.join(filter(str.isdigit, match_week_str))
                # Convert match_week to integer
                match_week = int(match_week)
                self.logger.debug(f"Section {section_idx}: Match Week: {match_week}, Date: {date}")
                return match_week, date
            except ValueError as e:
                self.logger.warning(
                    f"Section {section_idx}: Error parsing match week and date: {match_week_and_date}"
                )
                return None, None
        else:
            self.logger.warning(f"Section {section_idx}: No title div found for match week and date")
            return None, None
    
    def _extract_row_data(self, row, match_week, date, row_idx: int) -> list:
        """
        Extract data from a schedule row.
        
        Args:
            row: BeautifulSoup row element
            match_week: Match week number
            date: Match date string
            row_idx: Row index for logging
        
        Returns:
            List of row data
        """
        columns = row.find_all('div', recursive=False)
        row_data = [col.text.strip() for col in columns]
        
        # Ensure the correct number of columns (add empty result if missing)
        if len(row_data) == 5:  # Missing result
            row_data.append('')  # Add empty result
            self.logger.debug(f"Row {row_idx}: Missing result, added empty string")
        
        # Add match week and date to each row
        row_data.extend([match_week, date])
        self.logger.debug(f"Row {row_idx}: Extracted data: {row_data}")
        
        return row_data
    
    def _create_dataframe(self, data_rows: list) -> pd.DataFrame:
        """
        Create and clean DataFrame from scraped data rows.
        
        Args:
            data_rows: List of row data
        
        Returns:
            Cleaned DataFrame
        """
        column_names = ['Home Team', 'vs', 'Away Team', 'Venue', 'Time', 'Result', 'Match Week', 'Date']
        df = pd.DataFrame(data_rows, columns=column_names)
        self.logger.info(f"Successfully created schedules and results DataFrame with {len(df)} rows")
        
        # Convert 'Match Week' to numeric and handle NaN values
        df['Match Week'] = pd.to_numeric(df['Match Week'], errors='coerce')
        
        # Drop rows with NaN in 'Match Week' if necessary
        initial_row_count = len(df)
        df = df.dropna(subset=['Match Week'])
        self.logger.info(f"Dropped {initial_row_count - len(df)} rows with NaN in 'Match Week'")
        
        # Convert 'Match Week' to integer type
        df['Match Week'] = df['Match Week'].astype(int)
        
        return df
