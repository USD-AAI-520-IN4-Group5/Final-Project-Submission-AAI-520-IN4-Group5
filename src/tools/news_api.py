"""
News API client for fetching financial news using the News API service.
"""
import json
import requests
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.utils.config_loader import ConfigLoader
from src.utils.text_processing import (
    analyze_news_impact, classify_topic, is_political_content, 
    is_finance_relevant, preprocess_article
)

class NewsAPIClient:
    """
    News API client that fetches real financial news data from News API.
    """
    
    def __init__(self, api_key: str = None, config: Optional[ConfigLoader] = None, openai_api_key: Optional[str] = None):
        """
        Initialize the news API client.
        
        Args:
            api_key: API key for the News API service (optional if config provided)
            config: Configuration loader instance
            openai_api_key: OpenAI API key for advanced LLM analysis
        """
        self.config = config or ConfigLoader()
        self.api_key = api_key or self.config.get_news_api_key()
        self.openai_api_key = openai_api_key
        self.base_url = "https://newsapi.org/v2"
        self.headers = {
            "X-API-Key": self.api_key
        }
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make a request to the News API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return {"status": "error", "articles": []}
    
    def search_symbol_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for news articles related to a specific stock symbol with finance filtering.
        
        Args:
            symbol: Stock symbol to search for (e.g., 'AAPL')
            limit: Maximum number of articles to return
            
        Returns:
            List of filtered news articles with title, content, published_at, and url
        """
        # Search for company name and symbol
        company_name = self._get_company_name(symbol)
        
        # Build finance-specific query if filtering is enabled
        if self.config.is_finance_filter_enabled():
            query = self._build_finance_query(symbol, company_name)
        else:
            query = f"{symbol} OR {company_name}"
        
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit * 3, 100),  # Fetch more to account for filtering
            "from": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        }
        
        response = self._make_request("everything", params)
        
        if response.get("status") == "ok":
            articles = response.get("articles", [])
            # Format and filter articles
            formatted_articles = []
            for article in articles:
                formatted_article = {
                    "title": article.get("title", ""),
                    "content": article.get("description", "") or article.get("content", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "author": article.get("author", "")
                }
                
                # Apply finance filtering if enabled
                if self.config.is_finance_filter_enabled():
                    if self._is_finance_relevant(formatted_article):
                        formatted_articles.append(formatted_article)
                else:
                    formatted_articles.append(formatted_article)
            
            return formatted_articles[:limit]
        else:
            print(f"News API error: {response.get('message', 'Unknown error')}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """
        Get company name from stock symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company name
        """
        symbol_to_name = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "TSLA": "Tesla",
            "META": "Meta",
            "NVDA": "NVIDIA",
            "NFLX": "Netflix",
            "AMD": "AMD",
            "INTC": "Intel"
        }
        return symbol_to_name.get(symbol.upper(), symbol)
    
    def _build_finance_query(self, symbol: str, company_name: str) -> str:
        """
        Build a finance-specific search query using generic approach.
        
        Args:
            symbol: Stock symbol
            company_name: Company name
            
        Returns:
            Finance-focused search query
        """
        # Generic approach: Use symbol with financial context terms
        # Let the LLM-based filtering handle relevance
        query = f'"{symbol}" AND (financial OR business OR market OR investment OR corporate OR company OR stock OR trading)'
        return query
    
    def _is_finance_relevant(self, article: Dict[str, Any]) -> bool:
        """
        Check if an article is finance-relevant using LLM-based analysis.
        
        Args:
            article: Article dictionary with title, content, etc.
            
        Returns:
            True if article is finance-relevant
        """
        title = article.get("title", "")
        content = article.get("content", "")
        
        # Combine title and content for analysis
        text = f"{title} {content}"
        
        if not text.strip():
            return False
        
        # Use LLM-based finance relevance check only
        try:
            return is_finance_relevant(text, self.openai_api_key)
        except Exception as e:
            print(f"Error in LLM finance relevance check: {e}")
            # Generic fallback: exclude if LLM fails
            return False
    
    def get_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the latest financial news articles with finance filtering.
        
        Args:
            limit: Maximum number of articles to return
            
        Returns:
            List of filtered latest news articles
        """
        params = {
            "category": "business",
            "language": "en",
            "pageSize": min(limit * 3, 100),  # Fetch more to account for filtering
            "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        }
        
        response = self._make_request("top-headlines", params)
        
        if response.get("status") == "ok":
            articles = response.get("articles", [])
            # Format and filter articles
            formatted_articles = []
            for article in articles:
                formatted_article = {
                    "title": article.get("title", ""),
                    "content": article.get("description", "") or article.get("content", ""),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "author": article.get("author", "")
                }
                
                # Apply finance filtering if enabled
                if self.config.is_finance_filter_enabled():
                    if self._is_finance_relevant(formatted_article):
                        formatted_articles.append(formatted_article)
                else:
                    formatted_articles.append(formatted_article)
            
            return formatted_articles[:limit]
        else:
            print(f"News API error: {response.get('message', 'Unknown error')}")
            return []
