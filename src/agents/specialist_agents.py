"""
Specialist agents using LLM-based analysis instead of hardcoded keywords.
All analysis is now AI-driven for better accuracy and context understanding.
"""
import logging
from typing import List, Dict, Any, Optional
from src.utils.text_processing import analyze_news_impact, classify_topic, is_political_content

logger = logging.getLogger(__name__)

class EarningsAgent:
    """Earnings analysis agent using LLM-based approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
    
    def analyze(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze earnings-related signals using LLM approach."""
        signals = []
        
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            
            try:
                # Use LLM-based sentiment analysis
                sentiment_result = analyze_news_impact(text, self.openai_api_key)
                
                # Use LLM-based topic classification
                topic = classify_topic(text, self.openai_api_key)
                
                # Only process earnings-related content
                if topic == "EARNINGS":
                    signal_type = self._determine_earnings_signal_type(sentiment_result, text)
                    
                    if signal_type:
                        signals.append({
                            "type": signal_type,
                            "source": item,
                            "confidence": sentiment_result.get("confidence", 0.5),
                            "description": f"Earnings signal detected: {signal_type}",
                            "sentiment": sentiment_result.get("sentiment", "neutral"),
                            "score": sentiment_result.get("score", 0.0)
                        })
                
                # Extract numeric claims
                numbers = item.get("numbers", [])
                if numbers:
                    signals.append({
                        "type": "numeric_claim",
                        "numbers": numbers,
                        "source": item,
                        "confidence": 0.6,
                        "description": f"Numeric claims found: {', '.join(numbers)}"
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing earnings item: {e}")
                continue
        
        return signals
    
    def _determine_earnings_signal_type(self, sentiment_result: Dict[str, Any], text: str) -> Optional[str]:
        """Determine earnings signal type based on sentiment and context."""
        sentiment = sentiment_result.get("sentiment", "neutral")
        score = sentiment_result.get("score", 0.0)
        
        # Use sentiment and score to determine signal type
        if sentiment == "positive" and score > 0.3:
            return "earnings_beat"
        elif sentiment == "negative" and score < -0.3:
            return "earnings_miss"
        elif "guidance" in text.lower():
            if sentiment == "positive":
                return "guidance_raise"
            elif sentiment == "negative":
                return "guidance_cut"
        
        return None

class NewsImpactAgent:
    """News impact analysis agent using LLM-based approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
    
    def analyze_news_impact(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze news impact using LLM approach."""
        signals = []
        
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            
            try:
                # Use LLM-based sentiment analysis
                sentiment_result = analyze_news_impact(text, self.openai_api_key)
                
                # Determine impact level based on sentiment and confidence
                confidence = sentiment_result.get("confidence", 0.0)
                sentiment = sentiment_result.get("sentiment", "neutral")
                score = sentiment_result.get("score", 0.0)
                
                if confidence > 0.6:  # High confidence threshold
                    impact_level = "high" if abs(score) > 0.5 else "medium"
                    
                    signals.append({
                        "type": f"news_impact_{sentiment}",
                        "source": item,
                        "confidence": confidence,
                        "description": f"High-impact news detected: {sentiment} sentiment",
                        "impact_level": impact_level,
                        "sentiment": sentiment,
                        "score": score
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing news impact: {e}")
                continue
        
        return signals

class TechnicalAnalysisAgent:
    """Technical analysis agent using LLM-based approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
    
    def analyze(self, price_data: Dict[str, Any], items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze technical signals using LLM approach."""
        signals = []
        
        # Analyze price data for technical patterns
        if price_data and isinstance(price_data, dict):
            signals.extend(self._analyze_price_patterns(price_data))
        
        # Analyze news items for technical sentiment
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            
            try:
                # Use LLM-based sentiment analysis
                sentiment_result = analyze_news_impact(text, self.openai_api_key)
                
                # Look for technical analysis keywords in context
                if any(term in text.lower() for term in ["technical", "chart", "pattern", "resistance", "support", "trend"]):
                    signals.append({
                        "type": f"technical_sentiment_{sentiment_result.get('sentiment', 'neutral')}",
                        "source": item,
                        "confidence": sentiment_result.get("confidence", 0.5),
                        "description": f"Technical analysis sentiment: {sentiment_result.get('sentiment', 'neutral')}",
                        "sentiment": sentiment_result.get("sentiment", "neutral"),
                        "score": sentiment_result.get("score", 0.0)
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing technical signals: {e}")
                continue
        
        return signals
    
    def _analyze_price_patterns(self, price_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze price patterns for technical signals."""
        signals = []
        
        try:
            # Simple technical analysis based on price data
            if "close" in price_data and "open" in price_data:
                close_price = float(price_data.get("close", 0))
                open_price = float(price_data.get("open", 0))
                
                if close_price > open_price:
                    signals.append({
                        "type": "bullish_candle",
                        "confidence": 0.6,
                        "description": "Bullish candle pattern detected",
                        "price_change": close_price - open_price
                    })
                elif close_price < open_price:
                    signals.append({
                        "type": "bearish_candle",
                        "confidence": 0.6,
                        "description": "Bearish candle pattern detected",
                        "price_change": close_price - open_price
                    })
                    
        except Exception as e:
            logger.error(f"Error analyzing price patterns: {e}")
        
        return signals

class RegulatoryAgent:
    """Regulatory analysis agent using LLM-based approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
    
    def analyze(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze regulatory signals using LLM approach."""
        signals = []
        
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            
            try:
                # Use LLM-based topic classification
                topic = classify_topic(text, self.openai_api_key)
                
                # Only process regulatory-related content
                if topic == "REGULATORY":
                    # Use LLM-based sentiment analysis
                    sentiment_result = analyze_news_impact(text, self.openai_api_key)
                    
                    signal_type = self._determine_regulatory_signal_type(sentiment_result, text)
                    
                    if signal_type:
                        signals.append({
                            "type": signal_type,
                            "source": item,
                            "confidence": sentiment_result.get("confidence", 0.5),
                            "description": f"Regulatory signal detected: {signal_type}",
                            "sentiment": sentiment_result.get("sentiment", "neutral"),
                            "score": sentiment_result.get("score", 0.0)
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing regulatory signals: {e}")
                continue
        
        return signals
    
    def _determine_regulatory_signal_type(self, sentiment_result: Dict[str, Any], text: str) -> Optional[str]:
        """Determine regulatory signal type based on sentiment and context."""
        sentiment = sentiment_result.get("sentiment", "neutral")
        score = sentiment_result.get("score", 0.0)
        
        # Use sentiment to determine regulatory impact
        if sentiment == "negative" and score < -0.3:
            return "regulatory_concern"
        elif sentiment == "positive" and score > 0.3:
            return "regulatory_approval"
        elif "investigation" in text.lower() or "inquiry" in text.lower():
            return "regulatory_investigation"
        elif "fine" in text.lower() or "penalty" in text.lower():
            return "regulatory_penalty"
        
        return None

class CorporateGovernanceAgent:
    """Corporate governance analysis agent using LLM-based approach."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
    
    def analyze(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze corporate governance signals using LLM approach."""
        signals = []
        
        for item in items:
            text = item.get("text", "")
            if not text:
                continue
            
            try:
                # Use LLM-based topic classification
                topic = classify_topic(text, self.openai_api_key)
                
                # Only process corporate governance-related content
                if topic == "CORP_GOV":
                    # Use LLM-based sentiment analysis
                    sentiment_result = analyze_news_impact(text, self.openai_api_key)
                    
                    signal_type = self._determine_governance_signal_type(sentiment_result, text)
                    
                    if signal_type:
                        signals.append({
                            "type": signal_type,
                            "source": item,
                            "confidence": sentiment_result.get("confidence", 0.5),
                            "description": f"Corporate governance signal detected: {signal_type}",
                            "sentiment": sentiment_result.get("sentiment", "neutral"),
                            "score": sentiment_result.get("score", 0.0)
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing corporate governance signals: {e}")
                continue
        
        return signals
    
    def _determine_governance_signal_type(self, sentiment_result: Dict[str, Any], text: str) -> Optional[str]:
        """Determine corporate governance signal type based on sentiment and context."""
        sentiment = sentiment_result.get("sentiment", "neutral")
        score = sentiment_result.get("score", 0.0)
        
        # Use sentiment to determine governance impact
        if sentiment == "negative" and score < -0.3:
            return "governance_concern"
        elif sentiment == "positive" and score > 0.3:
            return "governance_positive"
        elif "ceo" in text.lower() or "leadership" in text.lower():
            if "resign" in text.lower() or "step down" in text.lower():
                return "leadership_change"
            elif "appoint" in text.lower() or "hire" in text.lower():
                return "leadership_appointment"
        
        return None