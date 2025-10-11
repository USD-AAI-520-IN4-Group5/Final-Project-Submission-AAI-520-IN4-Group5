"""
Earnings analysis agent using LLM-based approach.
Removes all hardcoded keyword-based analysis.
"""
import logging
from typing import List, Dict, Any, Optional
from src.utils.text_processing import analyze_news_impact, classify_topic

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
                # Use LLM-based topic classification
                topic = classify_topic(text, self.openai_api_key)
                
                # Only process earnings-related content
                if topic == "EARNINGS":
                    # Use LLM-based sentiment analysis
                    sentiment_result = analyze_news_impact(text, self.openai_api_key)
                    
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