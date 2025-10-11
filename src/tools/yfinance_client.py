"""
Yahoo Finance client for fetching stock data, financials, and technical indicators.
Based on the Exploration.ipynb and Research_Agent_Rule_Based.ipynb notebooks.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Suppress yfinance warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class YFinanceClient:
    """
    Yahoo Finance client with caching and technical analysis capabilities.
    """
    
    def __init__(self, cache_dir: str = "./yahoo_cache"):
        """
        Initialize the Yahoo Finance client.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _normalize_for_json(self, obj):
        """Normalize objects for JSON serialization."""
        if isinstance(obj, pd.DataFrame):
            return [self._normalize_for_json(r) for r in obj.reset_index().to_dict(orient="records")]
        if isinstance(obj, pd.Series):
            try:
                return self._normalize_for_json(obj.to_dict())
            except Exception:
                return [self._normalize_for_json(x) for x in obj.tolist()]
        if isinstance(obj, dict):
            return {str(k): self._normalize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._normalize_for_json(x) for x in list(obj)]
        if isinstance(obj, (pd.Timestamp, pd.Timedelta, pd.Interval)):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [self._normalize_for_json(x) for x in obj.tolist()]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.bool_):
            return bool(obj)
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
    
    def _cache_get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf8"))
            except Exception:
                return None
        return None
    
    def _cache_set(self, key: str, obj: Any):
        """Set cached data."""
        cache_file = self.cache_dir / f"{key}.json"
        safe_obj = self._normalize_for_json(obj)
        cache_file.write_text(
            json.dumps(safe_obj, ensure_ascii=False, indent=2), 
            encoding="utf8"
        )
    
    def _safe_str(self, x: Any, default: str = "") -> str:
        """Safely convert to string."""
        return str(x) if x is not None else default
    
    def get_ticker(self, ticker: str) -> yf.Ticker:
        """Get yfinance ticker object."""
        if not ticker:
            raise ValueError("Ticker must be provided")
        return yf.Ticker(str(ticker))
    
    def fetch_price_history(self, ticker: str, period: str = "1y", interval: str = "1d") -> List[Dict[str, Any]]:
        """
        Fetch price history with technical indicators.
        
        Args:
            ticker: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            List of price records with technical indicators
        """
        key = f"price_{ticker}_{period}_{interval}"
        cached = self._cache_get(key)
        if cached:
            return cached
        
        try:
            t = self.get_ticker(ticker)
            df = t.history(period=period, interval=interval)
            
            if df.empty:
                return []
            
            # Add technical indicators
            df["Daily_Return"] = df["Close"].pct_change()
            df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
            df["MA_20"] = df["Close"].rolling(20).mean()
            df["MA_50"] = df["Close"].rolling(50).mean()
            df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
            df["RSI"] = self._calculate_rsi(df["Close"])
            df["Bollinger_Upper"] = df["MA_20"] + (df["Volatility_20"] * 2)
            df["Bollinger_Lower"] = df["MA_20"] - (df["Volatility_20"] * 2)
            
            # Calculate annual metrics
            annual_return = df["Daily_Return"].mean() * 252
            annual_volatility = df["Daily_Return"].std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            records = df.reset_index().to_dict(orient="records")
            
            # Add summary metrics
            result = {
                "data": records,
                "summary": {
                    "annual_return": float(annual_return),
                    "annual_volatility": float(annual_volatility),
                    "sharpe_ratio": float(sharpe_ratio),
                    "current_price": float(df["Close"].iloc[-1]),
                    "price_change": float(df["Close"].iloc[-1] - df["Close"].iloc[-2]) if len(df) > 1 else 0,
                    "volume": int(df["Volume"].iloc[-1])
                }
            }
            
            self._cache_set(key, result)
            return result
            
        except Exception as e:
            print(f"Error fetching price history for {ticker}: {e}")
            return {"data": [], "summary": {}}
    
    def fetch_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch company information.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Company information dictionary
        """
        key = f"info_{ticker}"
        cached = self._cache_get(key)
        if cached:
            return cached
        
        try:
            t = self.get_ticker(ticker)
            info = getattr(t, "info", {}) or {}
            
            # Extract key metrics
            key_info = {
                "symbol": ticker,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "enterprise_value": info.get("enterpriseValue", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "beta": info.get("beta", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "avg_volume": info.get("averageVolume", 0),
                "employees": info.get("fullTimeEmployees", 0),
                "website": info.get("website", ""),
                "description": info.get("longBusinessSummary", "")
            }
            
            self._cache_set(key, key_info)
            return key_info
            
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return {}
    
    def fetch_financials(self, ticker: str, quarterly: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch financial statements.
        
        Args:
            ticker: Stock symbol
            quarterly: Whether to fetch quarterly data
            
        Returns:
            List of financial records
        """
        key = f"financials_{ticker}_{'q' if quarterly else 'a'}"
        cached = self._cache_get(key)
        if cached:
            return cached
        
        try:
            t = self.get_ticker(ticker)
            df = t.quarterly_financials if quarterly else t.financials
            
            if isinstance(df, pd.DataFrame) and not df.empty:
                records = df.reset_index().to_dict(orient="records")
            else:
                records = []
            
            self._cache_set(key, records)
            return records
            
        except Exception as e:
            print(f"Error fetching financials for {ticker}: {e}")
            return []
    
    def fetch_earnings(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch earnings data.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Earnings data dictionary
        """
        key = f"earnings_{ticker}"
        cached = self._cache_get(key)
        if cached:
            return cached
        
        try:
            t = self.get_ticker(ticker)
            
            # Try to get earnings data (deprecated but still works)
            earnings_df = getattr(t, "earnings", None)
            if isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty:
                earnings = earnings_df.reset_index().to_dict(orient="records")
            else:
                earnings = []
            
            # Get calendar data
            try:
                cal = getattr(t, "calendar", None)
                if hasattr(cal, "to_dict"):
                    calendar = cal.to_dict()
                else:
                    calendar = {}
            except Exception:
                calendar = {}
            
            result = {"earnings": earnings, "calendar": calendar}
            self._cache_set(key, result)
            return result
            
        except Exception as e:
            print(f"Error fetching earnings for {ticker}: {e}")
            return {"earnings": [], "calendar": {}}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_technical_signals(self, ticker: str) -> Dict[str, Any]:
        """
        Get technical analysis signals.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Technical signals dictionary
        """
        price_data = self.fetch_price_history(ticker, period="6mo")
        if not price_data.get("data"):
            return {}
        
        df = pd.DataFrame(price_data["data"])
        if df.empty:
            return {}
        
        # Ensure we have a date column
        date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        
        signals = {}
        
        if len(df) >= 20:
            # Moving average signals
            last_close = df["Close"].iloc[-1]
            ma20 = df["MA_20"].iloc[-1]
            ma50 = df["MA_50"].iloc[-1] if "MA_50" in df.columns else None
            
            signals["ma20_signal"] = "above" if last_close > ma20 else "below"
            signals["ma20_value"] = float(ma20)
            
            if ma50:
                signals["ma50_signal"] = "above" if last_close > ma50 else "below"
                signals["ma50_value"] = float(ma50)
            
            # Volatility signal
            if "Volatility_20" in df.columns:
                signals["volatility_20"] = float(df["Volatility_20"].iloc[-1])
            
            # RSI signal
            if "RSI" in df.columns:
                rsi = df["RSI"].iloc[-1]
                signals["rsi"] = float(rsi)
                if rsi > 70:
                    signals["rsi_signal"] = "overbought"
                elif rsi < 30:
                    signals["rsi_signal"] = "oversold"
                else:
                    signals["rsi_signal"] = "neutral"
            
            # Bollinger Bands
            if "Bollinger_Upper" in df.columns and "Bollinger_Lower" in df.columns:
                bb_upper = df["Bollinger_Upper"].iloc[-1]
                bb_lower = df["Bollinger_Lower"].iloc[-1]
                signals["bollinger_position"] = "upper" if last_close > bb_upper else "lower" if last_close < bb_lower else "middle"
                signals["bollinger_upper"] = float(bb_upper)
                signals["bollinger_lower"] = float(bb_lower)
        
        return signals
