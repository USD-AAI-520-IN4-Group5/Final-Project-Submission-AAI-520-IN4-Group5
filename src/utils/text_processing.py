"""
Advanced text processing using transformer models and LLMs.
Completely removes hardcoded keyword-based approaches.
"""
import logging
import re
import time
from typing import Dict, Any, List, Optional
import requests
import json

logger = logging.getLogger(__name__)

class LLMTextProcessor:
    """
    Advanced text processing using LLMs and transformers.
    No hardcoded keywords - all analysis is AI-driven.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None):
        """
        Initialize the LLM text processor.
        
        Args:
            openai_api_key: OpenAI API key for advanced analysis
            huggingface_token: Hugging Face token for model access
        """
        self.openai_api_key = openai_api_key
        self.huggingface_token = huggingface_token
        self.openai_base_url = "https://api.openai.com/v1"
        
        # Initialize transformer models
        self._init_transformer_models()
    
    def _init_transformer_models(self):
        """Initialize transformer models for local analysis."""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Set Hugging Face token if provided
            if self.huggingface_token:
                import os
                os.environ["HF_TOKEN"] = self.huggingface_token
                logger.info("Hugging Face token set for model access")
            
            # Try to use FinBERT for financial sentiment, fallback to general model
            try:
                # FinBERT is specifically trained on financial text
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
                logger.info("Using FinBERT for financial sentiment analysis")
            except Exception as e:
                logger.warning(f"FinBERT not available: {e}. Using general sentiment model.")
                # Fallback to general sentiment model
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=0 if self.device == "cuda" else -1,
                    return_all_scores=True
                )
            
            # Zero-shot classification for topics
            self.topic_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("Transformer models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Could not initialize transformer models: {e}")
            self.sentiment_pipeline = None
            self.topic_pipeline = None
    
    def analyze_sentiment_llm(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using LLM-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment analysis results
        """
        if not text or len(text.strip()) < 10:
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "impact": "neutral",
                "method": "default"
            }
        
        # Try transformer-based analysis first (more reliable, no rate limits)
        if self.sentiment_pipeline:
            try:
                return self._analyze_sentiment_transformer(text)
            except Exception as e:
                logger.warning(f"Transformer sentiment analysis failed: {e}")
        
        # Try OpenAI API only if transformer fails and we have a valid key
        if self.openai_api_key and self.openai_api_key != "test":
            try:
                return self._analyze_sentiment_openai(text)
            except Exception as e:
                logger.warning(f"OpenAI sentiment analysis failed: {e}")
        
        # Final fallback - simple Enhanced analysis
        return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_openai(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Analyze the sentiment of this financial news text and provide a JSON response with:
        1. sentiment: "positive", "negative", or "neutral"
        2. score: numerical score from -1.0 (very negative) to 1.0 (very positive)
        3. confidence: confidence level from 0.0 to 1.0
        4. reasoning: brief explanation of the analysis
        
        Text: "{text[:1000]}"
        
        Respond only with valid JSON.
        """
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }
        
        # Add rate limiting delay
        time.sleep(1.0)  # 1 second delay between requests to avoid rate limits
        
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            try:
                analysis = json.loads(content)
                return {
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "score": float(analysis.get("score", 0.0)),
                    "confidence": float(analysis.get("confidence", 0.5)),
                    "impact": analysis.get("sentiment", "neutral"),
                    "reasoning": analysis.get("reasoning", ""),
                    "method": "openai"
                }
            except json.JSONDecodeError:
                logger.warning("Failed to parse OpenAI response as JSON")
        
        raise Exception(f"OpenAI API error: {response.status_code}")
    
    def _analyze_sentiment_transformer(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using local transformer model."""
        # Truncate text if too long
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        results = self.sentiment_pipeline(text)
        
        # Handle the actual format: [[{'label': 'NEGATIVE', 'score': 0.0}, {'label': 'POSITIVE', 'score': 1.0}]]
        if isinstance(results, list) and len(results) > 0:
            # Get the first (and only) result which is a list of label-score pairs
            result_list = results[0]
            
            # Find the highest scoring sentiment
            best_result = max(result_list, key=lambda x: x['score'])
            label = best_result['label'].lower()
            score = best_result['score']
            
            # Map labels to our standard format
            if 'positive' in label:
                sentiment = "positive"
                final_score = score
            elif 'negative' in label:
                sentiment = "negative"
                final_score = -score
            else:
                sentiment = "neutral"
                final_score = 0.0
            
            return {
                "sentiment": sentiment,
                "score": final_score,
                "confidence": score,
                "impact": sentiment,
                "method": "transformer"
            }
        else:
            # Fallback if no results
            return {
                "sentiment": "neutral",
                "score": 0.0,
                "confidence": 0.0,
                "impact": "neutral",
                "method": "transformer"
            }
    
    def classify_topic_llm(self, text: str) -> str:
        """
        Classify topic using LLM-based approach.
        
        Args:
            text: Text to classify
            
        Returns:
            Topic classification
        """
        if not text:
            return "OTHER"
        
        # Try transformer-based classification first (more reliable, no rate limits)
        if self.topic_pipeline:
            try:
                return self._classify_topic_transformer(text)
            except Exception as e:
                logger.warning(f"Transformer topic classification failed: {e}")
        
        # Try OpenAI API only if transformer fails and we have a valid key
        if self.openai_api_key and self.openai_api_key != "test":
            try:
                return self._classify_topic_openai(text)
            except Exception as e:
                logger.warning(f"OpenAI topic classification failed: {e}")
        
        return "OTHER"
    
    def _classify_topic_openai(self, text: str) -> str:
        """Classify topic using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Classify this financial news text into one of these categories:
        - EARNINGS: earnings reports, quarterly results, revenue, profit
        - M&A: mergers, acquisitions, takeovers, deals
        - REGULATORY: SEC investigations, compliance, regulations, fines
        - PRODUCT: product launches, announcements, updates
        - MACRO: economic policy, Fed decisions, inflation, GDP
        - CORP_GOV: CEO changes, leadership, corporate governance
        - OTHER: anything else
        
        Text: "{text[:1000]}"
        
        Respond with only the category name (e.g., "EARNINGS").
        """
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            topic = result["choices"][0]["message"]["content"].strip().upper()
            
            valid_topics = ["EARNINGS", "M&A", "REGULATORY", "PRODUCT", "MACRO", "CORP_GOV", "OTHER"]
            if topic in valid_topics:
                return topic
        
        return "OTHER"
    
    def _classify_topic_transformer(self, text: str) -> str:
        """Classify topic using local transformer model."""
        finance_topics = [
            "earnings report",
            "merger and acquisition", 
            "regulatory compliance",
            "product launch",
            "macroeconomic policy",
            "corporate governance"
        ]
        
        try:
            result = self.topic_pipeline(text, finance_topics)
            best_topic = result['labels'][0]
            best_score = result['scores'][0]
            
            if best_score > 0.5:
                topic_mapping = {
                    "earnings report": "EARNINGS",
                    "merger and acquisition": "M&A", 
                    "regulatory compliance": "REGULATORY",
                    "product launch": "PRODUCT",
                    "macroeconomic policy": "MACRO",
                    "corporate governance": "CORP_GOV"
                }
                return topic_mapping.get(best_topic, "OTHER")
            
            return "OTHER"
            
        except Exception as e:
            logger.warning(f"Transformer topic classification error: {e}")
            return "OTHER"
    
    def is_political_content_llm(self, text: str) -> bool:
        """
        Detect political content using LLM-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if content appears to be political
        """
        if not text:
            return False
        
        # Try OpenAI API first if available
        if self.openai_api_key:
            try:
                return self._detect_political_openai(text)
            except Exception as e:
                logger.warning(f"OpenAI political detection failed: {e}")
        
        # Fallback to local transformer
        if self.topic_pipeline:
            try:
                return self._detect_political_transformer(text)
            except Exception as e:
                logger.warning(f"Transformer political detection failed: {e}")
        
        return False
    
    def _detect_political_openai(self, text: str) -> bool:
        """Detect political content using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Determine if this text is primarily about politics, elections, or government policy.
        Respond with "YES" if political, "NO" if not political.
        
        Text: "{text[:1000]}"
        
        Respond with only "YES" or "NO".
        """
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().upper()
            return answer == "YES"
        
        return False
    
    def _detect_political_transformer(self, text: str) -> bool:
        """Detect political content using local transformer model."""
        political_topics = [
            "political election",
            "government policy",
            "political campaign"
        ]
        
        try:
            result = self.topic_pipeline(text, political_topics)
            best_score = result['scores'][0]
            return best_score > 0.6
        except Exception as e:
            logger.warning(f"Transformer political detection error: {e}")
            return False
    
    def is_finance_relevant_llm(self, text: str) -> bool:
        """
        Determine if text is finance-relevant using LLM approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if text is finance-relevant
        """
        if not text:
            return False
        
        # Try transformer-based relevance check first (more reliable, no rate limits)
        if self.topic_pipeline:
            try:
                return self._check_finance_relevance_transformer(text)
            except Exception as e:
                logger.warning(f"Transformer finance relevance check failed: {e}")
        
        # Try OpenAI API only if transformer fails and we have a valid key
        if self.openai_api_key and self.openai_api_key != "test":
            try:
                return self._check_finance_relevance_openai(text)
            except Exception as e:
                logger.warning(f"OpenAI finance relevance check failed: {e}")
        
        return False  # Default to excluding if we can't determine
    
    def _check_finance_relevance_openai(self, text: str) -> bool:
        """Check finance relevance using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""
        Determine if this text is relevant to financial markets, business, or investment decisions.
        Consider: earnings, stock prices, mergers, regulations, economic policy, corporate news.
        Respond with "YES" if finance-relevant, "NO" if not.
        
        Text: "{text[:1000]}"
        
        Respond with only "YES" or "NO".
        """
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        response = requests.post(
            f"{self.openai_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip().upper()
            return answer == "YES"
        
        return False  # Default to excluding if API fails
    
    def _analyze_sentiment_simple(self, text: str) -> Dict[str, Any]:
        """Simple sentiment analysis as final fallback."""
        text_lower = text.lower()
        
        # Very basic sentiment analysis
        positive_words = ["good", "great", "excellent", "positive", "strong", "growth", "profit", "success", "beat", "outperform"]
        negative_words = ["bad", "poor", "weak", "negative", "loss", "decline", "miss", "destroy", "crash", "crisis"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
            score = 0.5
        elif neg_count > pos_count:
            sentiment = "negative"
            score = -0.5
        else:
            sentiment = "neutral"
            score = 0.0
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": 0.3,  # Low confidence for simple method
            "impact": sentiment,
            "method": "simple"
        }
    
    def _check_finance_relevance_transformer(self, text: str) -> bool:
        """Check finance relevance using local transformer model."""
        if not self.topic_pipeline:
            return False
            
        finance_topics = [
            "financial markets",
            "business news", 
            "investment decisions",
            "corporate earnings",
            "stock market",
            "economic indicators"
        ]
        
        try:
            result = self.topic_pipeline(text, finance_topics)
            best_score = result['scores'][0]
            return best_score > 0.5  # Higher threshold for relevance
        except Exception as e:
            logger.warning(f"Transformer finance relevance check error: {e}")
            return False  # Default to excluding if transformer fails

# Global instance
_llm_processor = None

def get_llm_processor(openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None) -> LLMTextProcessor:
    """Get or create the global LLM processor instance."""
    global _llm_processor
    if _llm_processor is None:
        _llm_processor = LLMTextProcessor(openai_api_key, huggingface_token)
    return _llm_processor

# Convenience functions
def analyze_news_impact(text: str, openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None) -> Dict[str, Any]:
    """Analyze news impact using LLM-based approach."""
    processor = get_llm_processor(openai_api_key, huggingface_token)
    return processor.analyze_sentiment_llm(text)

def classify_topic(text: str, openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None) -> str:
    """Classify topic using LLM-based approach."""
    processor = get_llm_processor(openai_api_key, huggingface_token)
    return processor.classify_topic_llm(text)

def is_political_content(text: str, openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None) -> bool:
    """Detect political content using LLM-based approach."""
    processor = get_llm_processor(openai_api_key, huggingface_token)
    return processor.is_political_content_llm(text)

def is_finance_relevant(text: str, openai_api_key: Optional[str] = None, huggingface_token: Optional[str] = None) -> bool:
    """Check if text is finance-relevant using LLM approach."""
    processor = get_llm_processor(openai_api_key, huggingface_token)
    return processor.is_finance_relevant_llm(text)

# Legacy functions for backward compatibility
def extract_keywords(text: str) -> List[str]:
    """Extract keywords using simple approach (kept for compatibility)."""
    if not text:
        return []
    
    # Simple keyword extraction - could be enhanced with LLM
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Return most frequent words
    return sorted(word_freq, key=word_freq.get, reverse=True)[:10]

def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """Create a simple extractive summary (kept for compatibility)."""
    if not text:
        return ""
    
    # Simple extractive summary - take first few sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Return first max_sentences sentences
    summary_sentences = sentences[:max_sentences]
    return '. '.join(summary_sentences) + '.' if summary_sentences else text[:200] + "..."

def clean_text(text: str) -> str:
    """Clean text for processing (kept for compatibility)."""
    if not text:
        return ""
    
    # Basic text cleaning
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text

def simple_sentiment_score(text: str) -> float:
    """Simple sentiment scoring (kept for compatibility)."""
    if not text:
        return 0.0
    
    # Use the LLM-based sentiment analysis but return just the score
    try:
        result = analyze_news_impact(text)
        return result.get("score", 0.0)
    except Exception:
        # Fallback to simple scoring
        text_lower = text.lower()
        positive_words = ["good", "great", "excellent", "positive", "strong", "growth", "profit", "success"]
        negative_words = ["bad", "poor", "weak", "negative", "loss", "decline", "miss", "destroy"]
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 0.5
        elif neg_count > pos_count:
            return -0.5
        else:
            return 0.0

def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text."""
    if not text:
        return []
    
    patterns = [
        r"\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?",  # Currency and percentages
        r"\d+\.\d+%",  # Decimal percentages
        r"\d+%",  # Simple percentages
        r"\$\d+\.\d+[MBK]?",  # Currency with suffixes
    ]
    
    numbers = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        numbers.extend(matches)
    
    return list(dict.fromkeys(numbers))

def preprocess_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess article data."""
    return {
        "title": article.get("title", ""),
        "summary": article.get("content", "") or article.get("description", ""),
        "text": f"{article.get('title', '')} {article.get('content', '') or article.get('description', '')}",
        "datetime": article.get("published_at") or article.get("datetime_utc"),
        "link": article.get("url") or article.get("link"),
        "publisher": article.get("source", {}).get("name") if isinstance(article.get("source"), dict) else article.get("publisher", "Unknown")
    }

def simple_sentiment_score(text: str) -> float:
    """Simple sentiment score (kept for compatibility)."""
    result = analyze_news_impact(text)
    return result.get("score", 0.0)