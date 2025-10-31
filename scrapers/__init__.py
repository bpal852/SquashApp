"""
Scrapers package for HK Squash League data extraction.

This package contains modular scraper classes for different pages of the HK Squash website.
Each scraper is responsible for extracting specific types of data.
"""

from .players import scrape_players_page
from .ranking import scrape_ranking_page
from .schedules import scrape_schedules_and_results_page
from .summary import scrape_team_summary_page
from .teams import scrape_teams_page

# Aliases for convenience
scrape_summary_page = scrape_team_summary_page

__all__ = [
    "scrape_teams_page",
    "scrape_team_summary_page",
    "scrape_summary_page",
    "scrape_schedules_and_results_page",
    "scrape_ranking_page",
    "scrape_players_page",
]
