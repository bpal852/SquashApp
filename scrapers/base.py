"""
Base scraper functionality for HK Squash website.

Provides common utilities and configuration for all scrapers.
"""

import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from typing import Optional


class BaseScraper:
    """Base class for all scrapers with common functionality."""
    
    BASE_URL = "https://www.hksquash.org.hk/public/index.php/leagues"
    PAGES_ID = "25"
    REQUEST_TIMEOUT = (10, 30)  # (connect timeout, read timeout)
    
    def __init__(self, session: Optional[requests.Session] = None):
        """
        Initialize the base scraper.
        
        Args:
            session: Optional requests.Session to use. If None, creates a new one.
        """
        self.session = session or self._build_session()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def _build_session() -> requests.Session:
        """
        Build a requests session with retry logic.
        
        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
    
    def build_url(self, path: str, league_id: str, year: str) -> str:
        """
        Build a URL for the HK Squash website.
        
        Args:
            path: The page path (e.g., "teams", "ranking")
            league_id: The league ID (e.g., "D00473")
            year: The year string (e.g., "2025-2026")
        
        Returns:
            Complete URL string
        """
        return (f"{self.BASE_URL}/{path}/id/{league_id}/"
                f"league/Squash/year/{year}/pages_id/{self.PAGES_ID}.html")
    
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a page from the HK Squash website.
        
        Args:
            url: The URL to fetch
        
        Returns:
            BeautifulSoup object if successful, None otherwise
        
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        self.logger.debug(f"Fetching URL: {url}")
        
        try:
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            self.logger.debug(f"Response status code: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"Failed to retrieve page. Status code: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            self.logger.debug("Successfully parsed HTML content")
            return soup
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Request timed out for URL: {url}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for URL: {url}. Error: {e}")
            raise
    
    def log_scrape_start(self, page_type: str, division: str, league_id: str, year: str):
        """Log the start of a scraping operation."""
        self.logger.info(f"Scraping {page_type} page for Division {division}")
        self.logger.info(f"Scraping {page_type} page for league id: {league_id}, year: {year}")
    
    def log_scrape_success(self, page_type: str, division: str, row_count: int):
        """Log successful completion of a scraping operation."""
        self.logger.info(f"Successfully scraped {page_type} page for Division {division}")
        self.logger.info(f"Created {page_type} DataFrame with {row_count} rows")
    
    def log_scrape_error(self, page_type: str, division: str, error: Exception):
        """Log an error during scraping operation."""
        self.logger.error(f"Error scraping {page_type} page for Division {division}: {error}")
        self.logger.exception(error)
