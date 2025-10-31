"""
Team Summary page scraper for HK Squash website.

Scrapes team summary information including matches played, won, lost, and points.
Handles both spellings of the page: 'team_summery' and 'team_summary'.
"""

from typing import Optional

import pandas as pd

from .base import BaseScraper


def scrape_team_summary_page(league_id: str, year: str, session=None) -> pd.DataFrame:
    """
    Scrape the Team Summary page from the HK Squash website.

    Tries both possible spellings of the URL:
    - team_summery (current spelling)
    - team_summary (fallback)

    Args:
        league_id: The league ID (e.g., "D00473")
        year: The year string (e.g., "2025-2026")
        session: Optional requests.Session to use

    Returns:
        DataFrame with columns: Team, Played, Won, Lost, Points
        Raises SystemExit(1) if both spellings fail
    """
    scraper = TeamSummaryScraper(session)
    return scraper.scrape(league_id, year)


class TeamSummaryScraper(BaseScraper):
    """Scraper for the team summary page."""

    SUMMARY_PATHS = ["team_summery", "team_summary"]

    def scrape(self, league_id: str, year: str) -> pd.DataFrame:
        """
        Scrape team summary data for a specific league and year.

        Tries both possible URL spellings and returns data from the first successful one.

        Args:
            league_id: The league ID (e.g., "D00473")
            year: The year string (e.g., "2025-2026")

        Returns:
            DataFrame with team summary information

        Raises:
            SystemExit: If both URL spellings fail
        """
        last_error = None

        for path in self.SUMMARY_PATHS:
            try:
                df = self._try_scrape_with_path(path, league_id, year)
                if df is not None and not df.empty:
                    return df
                last_error = ValueError(f"[{path}] parsed 0 data rows")
            except Exception as e:
                self.logger.exception(f"[{path}] Error scraping team summary: {e}")
                last_error = e
                continue

        # If we get here, both spellings failed
        self.logger.error(f"Team summary failed with both slugs for {league_id}: {last_error}")
        raise SystemExit(1)

    def _try_scrape_with_path(self, path: str, league_id: str, year: str) -> Optional[pd.DataFrame]:
        """
        Try to scrape team summary data using a specific URL path.

        Args:
            path: The URL path ("team_summery" or "team_summary")
            league_id: The league ID
            year: The year string

        Returns:
            DataFrame if successful, None if no data found
        """
        summary_url = self.build_url(path, league_id, year)
        self.logger.info(f"Scraping team summary page ({path}) for league id: {league_id}, year: {year}...")

        # Fetch page
        soup = self.fetch_page(summary_url)
        if soup is None:
            return None

        # Find summary rows using multiple selectors
        rows = (
            soup.select("div.clearfix.teamSummary-content-list")
            or soup.select("div.teamSummary-content-list")
            or soup.select("div.teamSummary div[class*='content-list']")
        )

        # Extract data from rows
        data = []
        for idx, row in enumerate(rows):
            cells = [d.get_text(strip=True) for d in row.find_all("div", recursive=False)]
            cells = [c for c in cells if c]

            # Skip header-like rows
            joined = "".join(cells).lower()
            if "played" in joined and "won" in joined and "lost" in joined:
                continue

            # Parse data if we have enough columns
            if len(cells) >= 5:
                # Team name might span multiple columns before the last 4 numeric columns
                team = " ".join(cells[:-4]) if len(cells) > 5 else cells[0]
                tail = cells[-4:]

                try:
                    p, w, l, pts = map(int, tail)
                    if team and not team.lower().startswith("played"):
                        data.append([team, p, w, l, pts])
                except Exception:
                    # Ignore malformed lines
                    pass

        # Check if we got any data
        if not data:
            return None

        # Create DataFrame
        df = pd.DataFrame(data, columns=["Team", "Played", "Won", "Lost", "Points"])
        df[["Played", "Won", "Lost", "Points"]] = (
            df[["Played", "Won", "Lost", "Points"]].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        )

        self.logger.info(f"[{path}] Successfully created summary DataFrame with {len(df)} rows")

        return df
