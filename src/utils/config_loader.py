"""
Configuration loader for Financial Agentic AI System.
Handles loading and validation of configuration settings.
"""
import yaml
import os
from typing import Dict, Any, List
from pathlib import Path

class ConfigLoader:
    """Configuration loader for the Financial Agentic AI system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Warning: Configuration file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return self._validate_config(config)
        except Exception as e:
            print(f"Error loading configuration: {e}. Using defaults.")
            return self._get_default_config()
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize configuration."""
        # Ensure all required sections exist
        if 'news_filtering' not in config:
            config['news_filtering'] = {}
        if 'analysis' not in config:
            config['analysis'] = {}
        if 'api' not in config:
            config['api'] = {}
        
        # Set defaults for missing values
        news_config = config['news_filtering']
        news_config.setdefault('enable_finance_filter', True)
        news_config.setdefault('llm_based_filtering', True)
        news_config.setdefault('exclude_keywords', [])
        news_config.setdefault('preferred_sources', [])
        news_config.setdefault('min_finance_relevance', 0.3)
        news_config.setdefault('enable_topic_filter', True)
        news_config.setdefault('allowed_topics', ['EARNINGS', 'M&A', 'REGULATORY', 'PRODUCT', 'MACRO', 'CORP_GOV'])
        
        analysis_config = config['analysis']
        analysis_config.setdefault('max_iterations', 2)
        analysis_config.setdefault('news_limit', 20)
        analysis_config.setdefault('confidence_threshold', 0.5)
        analysis_config.setdefault('enable_earnings_analysis', True)
        analysis_config.setdefault('enable_news_impact_analysis', True)
        analysis_config.setdefault('enable_technical_analysis', True)
        analysis_config.setdefault('enable_regulatory_analysis', True)
        analysis_config.setdefault('enable_corporate_governance_analysis', True)
        
        api_config = config['api']
        api_config.setdefault('news_api_key', "f8d9c4dbbb514174bb84682315ae9fa7")
        api_config.setdefault('request_timeout', 30)
        api_config.setdefault('max_retries', 3)
        
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'news_filtering': {
                'enable_finance_filter': True,
                'llm_based_filtering': True,
                'exclude_keywords': [
                    'sports', 'entertainment', 'celebrity', 'movie', 'music',
                    'gaming', 'weather', 'politics', 'election', 'crime',
                    'accident', 'health', 'medical'
                ],
                'preferred_sources': [
                    'bloomberg', 'reuters', 'wsj', 'financial times', 'cnbc',
                    'marketwatch', 'yahoo finance', 'seeking alpha', 'benzinga'
                ],
                'min_finance_relevance': 0.3,
                'enable_topic_filter': True,
                'allowed_topics': ['EARNINGS', 'M&A', 'REGULATORY', 'PRODUCT', 'MACRO', 'CORP_GOV']
            },
            'analysis': {
                'max_iterations': 2,
                'news_limit': 20,
                'confidence_threshold': 0.5,
                'enable_earnings_analysis': True,
                'enable_news_impact_analysis': True,
                'enable_technical_analysis': True,
                'enable_regulatory_analysis': True,
                'enable_corporate_governance_analysis': True
            },
            'api': {
                'news_api_key': "f8d9c4dbbb514174bb84682315ae9fa7",
                'request_timeout': 30,
                'max_retries': 3
            }
        }
    
    def get_news_filtering_config(self) -> Dict[str, Any]:
        """Get news filtering configuration."""
        return self.config.get('news_filtering', {})
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config.get('analysis', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.config.get('api', {})
    
    def is_finance_filter_enabled(self) -> bool:
        """Check if finance filtering is enabled."""
        return self.get_news_filtering_config().get('enable_finance_filter', True)
    
    def is_llm_based_filtering_enabled(self) -> bool:
        """Check if LLM-based filtering is enabled."""
        return self.get_news_filtering_config().get('llm_based_filtering', True)
    
    def get_exclude_keywords(self) -> List[str]:
        """Get keywords to exclude."""
        return self.get_news_filtering_config().get('exclude_keywords', [])
    
    def get_preferred_sources(self) -> List[str]:
        """Get preferred news sources."""
        return self.get_news_filtering_config().get('preferred_sources', [])
    
    def get_min_finance_relevance(self) -> float:
        """Get minimum finance relevance score."""
        return self.get_news_filtering_config().get('min_finance_relevance', 0.3)
    
    def get_allowed_topics(self) -> List[str]:
        """Get allowed topics for filtering."""
        return self.get_news_filtering_config().get('allowed_topics', [])
    
    def get_news_limit(self) -> int:
        """Get news limit for analysis."""
        return self.get_analysis_config().get('news_limit', 20)
    
    def get_max_iterations(self) -> int:
        """Get maximum iterations for analysis."""
        return self.get_analysis_config().get('max_iterations', 2)
    
    def get_news_api_key(self) -> str:
        """Get News API key."""
        return self.get_api_config().get('news_api_key', "")
    
    def get_openai_api_key(self) -> str:
        """Get OpenAI API key."""
        return self.get_api_config().get('openai_api_key', "")
    
    def get_huggingface_token(self) -> str:
        """Get Hugging Face token."""
        return self.get_api_config().get('huggingface_token', "")
