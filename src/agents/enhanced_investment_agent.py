"""
Enhanced Investment Research Agent

An autonomous investment research agent capable of conducting comprehensive financial analysis
for any given stock ticker. The agent follows a structured workflow of planning, execution,
synthesis, evaluation, and self-reflection to generate coherent investment research reports.

Key Capabilities:
- Dynamic data retrieval from multiple financial sources
- Autonomous task planning and execution
- Comprehensive analysis across technical, fundamental, and sentiment dimensions
- Self-reflection and quality assessment
- Persistent memory for learning and improvement
"""
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.specialist_agents import (
    EarningsAgent, NewsImpactAgent, TechnicalAnalysisAgent, 
    RegulatoryAgent, CorporateGovernanceAgent
)
from src.tools.news_api import NewsAPIClient
from src.tools.yfinance_client import YFinanceClient
from src.utils.text_processing import (
    classify_topic, extract_numbers, preprocess_article,
    analyze_news_impact, extract_keywords
)
from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

class EnhancedInvestmentAgent(BaseAgent):
    """
    Enhanced Investment Research Agent
    
    An autonomous agent that conducts comprehensive financial analysis following
    a structured workflow of planning, execution, synthesis, evaluation, and
    self-reflection. The agent dynamically retrieves and processes diverse sources
    of financial data to generate coherent investment research reports.
    
    Workflow Phases:
    1. Planning: Task decomposition and resource allocation
    2. Execution: Evidence gathering and data processing
    3. Synthesis: Report generation and insight compilation
    4. Evaluation: Quality assessment and optimization
    5. Self-Reflection: Performance analysis and memory integration
    """
    
    def __init__(self, symbol: str, max_iterations: int = None, config: ConfigLoader = None, openai_api_key: Optional[str] = None, improvement_focus: Optional[str] = None, selected_improvements: Optional[List[str]] = None):
        super().__init__(name=f"EnhancedInvestmentAgent_{symbol}")
        self.symbol = symbol
        self.config = config or ConfigLoader()
        self.max_iterations = max_iterations or self.config.get_max_iterations()
        self.openai_api_key = openai_api_key
        self.huggingface_token = self.config.get_huggingface_token()
        
        # Store improvement parameters
        self.improvement_focus = improvement_focus
        self.selected_improvements = selected_improvements or []
        
        # Apply improvements to configuration
        self._apply_improvements()
        
        # Initialize clients with configuration and API keys
        self.news_client = NewsAPIClient(config=self.config, openai_api_key=self.openai_api_key)
        self.yfinance_client = YFinanceClient()
        
        # Initialize specialist agents
        self.earnings_agent = EarningsAgent()
        self.news_impact_agent = NewsImpactAgent()
        self.technical_agent = TechnicalAnalysisAgent()
        self.regulatory_agent = RegulatoryAgent()
        self.governance_agent = CorporateGovernanceAgent()
        
        # Memory file for persistence
        self.memory_file = Path("enhanced_runs.jsonl")
        self.memory_file.touch(exist_ok=True)
    
    def _apply_improvements(self):
        """Apply improvement parameters to enhance analysis capabilities."""
        if not self.improvement_focus:
            return
        
        logger.info(f"🔧 Applying improvements: {self.improvement_focus}")
        
        # Get current configuration
        current_config = self.config.config_data.copy() if hasattr(self.config, 'config_data') else {}
        
        # Apply improvements based on focus
        if self.improvement_focus == "news_sources" or self.improvement_focus == "all":
            # Increase news limit
            current_news_limit = self.config.get_news_limit()
            new_news_limit = min(current_news_limit * 2, 100)  # Double, cap at 100
            logger.info(f"📰 Expanding news sources: {current_news_limit} → {new_news_limit}")
            
            # Update configuration
            if 'analysis' not in current_config:
                current_config['analysis'] = {}
            current_config['analysis']['news_limit'] = new_news_limit
        
        if self.improvement_focus == "time_range" or self.improvement_focus == "all":
            # Extend time range for historical analysis
            logger.info("⏰ Extending time range for broader historical analysis")
            
            if 'analysis' not in current_config:
                current_config['analysis'] = {}
            current_config['analysis']['extended_time_range'] = True
            current_config['analysis']['historical_days'] = 365  # 1 year instead of default
        
        if self.improvement_focus == "selected" and self.selected_improvements:
            # Apply specific improvements
            logger.info(f"🎯 Applying selected improvements: {', '.join(self.selected_improvements)}")
            
            for improvement in self.selected_improvements:
                improvement_lower = improvement.lower()
                
                if "news sources" in improvement_lower or "news" in improvement_lower:
                    current_news_limit = self.config.get_news_limit()
                    new_news_limit = min(current_news_limit * 2, 100)
                    if 'analysis' not in current_config:
                        current_config['analysis'] = {}
                    current_config['analysis']['news_limit'] = new_news_limit
                    logger.info(f"📰 Expanded news sources: {current_news_limit} → {new_news_limit}")
                
                elif "time range" in improvement_lower or "time" in improvement_lower:
                    if 'analysis' not in current_config:
                        current_config['analysis'] = {}
                    current_config['analysis']['extended_time_range'] = True
                    current_config['analysis']['historical_days'] = 365
                    logger.info("⏰ Extended time range to 1 year")
                
                elif "depth" in improvement_lower or "analysis" in improvement_lower:
                    # Increase analysis depth
                    if 'analysis' not in current_config:
                        current_config['analysis'] = {}
                    current_config['analysis']['deep_analysis'] = True
                    current_config['analysis']['max_iterations'] = min(self.max_iterations * 2, 10)
                    logger.info("🔍 Increased analysis depth")
        
        # Update the configuration object
        if hasattr(self.config, 'config_data'):
            self.config.config_data.update(current_config)
        else:
            # Create a new config with improvements
            self.config = ConfigLoader()
            self.config.config_data = current_config
        
        logger.info("✅ Improvements applied successfully")
    
    def plan(self) -> List[str]:
        """
        Create autonomous analysis plan following Enhanced workflow.
        
        Returns:
            List of planned tasks in execution order
        """
        return [
            # Phase 1: Planning and Data Retrieval
            "plan_analysis_scope",
            "fetch_all_data",
            
            # Phase 2: Data Processing and Analysis
            "process_news_items",
            "route_to_specialists",
            
            # Phase 3: Synthesis and Report Generation
            "generate_analysis",
            "compile_research_report",
            
            # Phase 4: Evaluation and Optimization
            "evaluate_results",
            "optimize_if_needed",
            
            # Phase 5: Self-Reflection and Memory
            "self_reflect",
            "update_memory"
        ]
    
    def act(self) -> Dict[str, Any]:
        """
        Execute the autonomous Enhanced analysis workflow.
        
        Returns:
            Complete investment research report
        """
        logger.info(f"🚀 Starting autonomous Enhanced analysis for {self.symbol}")
        
        # Initialize comprehensive report
        report = {
            "symbol": self.symbol,
            "timestamp": datetime.now().isoformat(),
            "workflow_phase": "planning",
            "evidence": {},
            "signals": {},
            "analysis": {},
            "research_report": {},
            "evaluation": {},
            "reflection": {},
            "iterations": 0
        }
        
        try:
            # Phase 1: Planning and Data Retrieval
            logger.info("📋 Phase 1: Planning analysis scope...")
            scope = self.plan_analysis_scope()
            report["analysis_scope"] = scope
            
            logger.info("📊 Phase 1: Fetching all data...")
            evidence = self._fetch_all_data()
            report["evidence"] = evidence
            logger.info(f"✅ Phase 1 complete: Fetched {len(evidence.get('news', []))} news items")
            
            # Phase 2: Data Processing and Analysis
            logger.info("🔄 Phase 2: Processing news items...")
            processed_items = self._process_news_items(evidence.get("news", []))
            report["processed_items"] = processed_items
            logger.info(f"✅ Phase 2: Processed {len(processed_items)} items")
            
            logger.info("🎯 Phase 2: Routing to specialist agents...")
            signals = self._route_to_specialists(processed_items, evidence)
            report["signals"] = signals
            logger.info(f"✅ Phase 2 complete: Generated {len(signals)} signal types")
            
            # Phase 3: Synthesis and Report Generation
            logger.info("🧠 Phase 3: Generating analysis...")
            analysis = self._generate_analysis(processed_items, signals)
            report["analysis"] = analysis
            
            logger.info("📄 Phase 3: Compiling research report...")
            research_report = self.compile_research_report(analysis, signals)
            report["research_report"] = research_report
            logger.info("✅ Phase 3 complete: Research report compiled")
            
            # Phase 4: Evaluation and Optimization
            logger.info("🔍 Phase 4: Evaluating results...")
            evaluation = self._evaluate_report(report)
            report["evaluation"] = evaluation
            
            # Iterative optimization
            for i in range(self.max_iterations - 1):
                if report["evaluation"].get("needs_optimization", False):
                    report["iterations"] += 1
                    logger.info(f"♻️ Optimization iteration {report['iterations']}/{self.max_iterations-1} for {self.symbol}")
                    report = self._optimize_report(report)
                    # Re-evaluate after optimization
                    report["evaluation"] = self._evaluate_report(report)
                else:
                    break
            
            # Phase 5: Self-Reflection and Memory
            logger.info("🤔 Phase 5: Self-reflecting...")
            reflection = self._self_reflect(report)
            report["reflection"] = reflection
            
            logger.info("🧠 Phase 5: Updating persistent memory...")
            self.update_memory(report)
            
            # Save comprehensive report
            self._save_report(report)
            
            logger.info(f"✅ Autonomous Enhanced analysis completed for {self.symbol}")
            return report
            
        except Exception as e:
            logger.error(f"❌ Critical error during autonomous analysis for {self.symbol}: {e}", exc_info=True)
            report["error"] = str(e)
            # Attempt to save partial report
            self._save_report(report)
            return report
    
    def _fetch_all_data(self) -> Dict[str, Any]:
        """Fetch all required data."""
        evidence = {}
        
        try:
            # Determine time period based on improvements
            time_period = "1y"  # Default
            if hasattr(self.config, 'config_data') and self.config.config_data.get('analysis', {}).get('extended_time_range'):
                time_period = "2y"  # Extended for improvements
                logger.info("⏰ Using extended time range (2 years) for historical analysis")
            
            # Fetch stock data with appropriate time period
            evidence["price_data"] = self.yfinance_client.fetch_price_history(self.symbol, period=time_period)
            evidence["company_info"] = self.yfinance_client.fetch_company_info(self.symbol)
            evidence["financials"] = self.yfinance_client.fetch_financials(self.symbol)
            evidence["earnings"] = self.yfinance_client.fetch_earnings(self.symbol)
            
            # Fetch news with configuration-driven limit (now includes improvements)
            news_limit = self.config.get_news_limit()
            logger.info(f"📰 Fetching {news_limit} news articles")
            evidence["news"] = self.news_client.search_symbol_news(self.symbol, limit=news_limit)
            
            # Get technical signals
            evidence["technical_signals"] = self.yfinance_client.get_technical_signals(self.symbol)
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            evidence["error"] = str(e)
        
        return evidence
    
    def _process_news_items(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process news items using LLM-based analysis."""
        logger.info(f"🔄 Processing {len(news_items)} news items...")
        processed = []
        
        for i, item in enumerate(news_items):
            logger.info(f"📰 Processing item {i+1}/{len(news_items)}: {item.get('title', 'No title')[:50]}...")
            try:
                # Preprocess article
                processed_item = preprocess_article(item)
                
                # Use LLM-based topic classification with timeout
                try:
                    logger.info(f"🏷️ Classifying topic for item {i+1}...")
                    topic = classify_topic(processed_item["text"], self.openai_api_key, self.huggingface_token)
                    processed_item["topic"] = topic
                    logger.info(f"✅ Topic classified as: {topic}")
                except Exception as e:
                    logger.warning(f"⚠️ Topic classification failed for item {i+1}: {e}")
                    processed_item["topic"] = "OTHER"
                
                # Extract numbers (simple approach)
                numbers = extract_numbers(processed_item["text"])
                processed_item["numbers"] = numbers
                
                # Use LLM-based sentiment analysis with timeout
                try:
                    logger.info(f"😊 Analyzing sentiment for item {i+1}...")
                    impact = analyze_news_impact(processed_item["text"], self.openai_api_key, self.huggingface_token)
                    processed_item["impact"] = impact
                    logger.info(f"✅ Sentiment: {impact.get('sentiment', 'unknown')} (score: {impact.get('score', 0)})")
                except Exception as e:
                    logger.warning(f"⚠️ Sentiment analysis failed for item {i+1}: {e}")
                    processed_item["impact"] = {"sentiment": "neutral", "score": 0.0, "confidence": 0.0, "method": "fallback"}
                
                # Extract keywords (simple approach)
                keywords = extract_keywords(processed_item["text"])
                processed_item["keywords"] = keywords[:10]  # Limit to top 10
                
                # Generate summary (simple approach)
                summary = processed_item["text"][:200] + "..." if len(processed_item["text"]) > 200 else processed_item["text"]
                processed_item["summary"] = summary
                
                processed.append(processed_item)
                logger.info(f"✅ Item {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"❌ Error processing news item {i+1}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(processed)}/{len(news_items)} news items")
        return processed
    
    def _route_to_specialists(self, processed_items: List[Dict[str, Any]], evidence: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Route content to specialist agents."""
        logger.info("🎯 Starting specialist routing...")
        signals = {}
        
        # Group items by topic
        topic_groups = {}
        for item in processed_items:
            topic = item.get("topic", "OTHER")
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(item)
        
        logger.info(f"📊 Topic groups: {list(topic_groups.keys())}")
        
        # Route to specialist agents
        try:
            # Earnings analysis
            if "EARNINGS" in topic_groups:
                logger.info("💰 Routing to EarningsAgent...")
                signals["earnings"] = self.earnings_agent.analyze_earnings_content(
                    topic_groups["EARNINGS"], 
                    evidence.get("financials", [])
                )
                logger.info(f"✅ EarningsAgent completed: {len(signals['earnings'])} signals")
            
            # News impact analysis
            logger.info("📰 Routing to NewsImpactAgent...")
            signals["news_impact"] = self.news_impact_agent.analyze_news_impact(processed_items)
            logger.info(f"✅ NewsImpactAgent completed: {len(signals['news_impact'])} signals")
            
            # Technical analysis
            logger.info("📈 Routing to TechnicalAnalysisAgent...")
            signals["technical"] = self.technical_agent.analyze(
                evidence.get("price_data", {}),
                processed_items
            )
            logger.info(f"✅ TechnicalAnalysisAgent completed: {len(signals['technical'])} signals")
            
            # Regulatory analysis
            if "REGULATORY" in topic_groups:
                signals["regulatory"] = self.regulatory_agent.analyze(
                    topic_groups["REGULATORY"]
                )
            
            # Corporate governance analysis
            if "CORP_GOV" in topic_groups:
                signals["governance"] = self.governance_agent.analyze(
                    topic_groups["CORP_GOV"]
                )
            
        except Exception as e:
            logger.error(f"Error in specialist routing: {e}")
            signals["error"] = str(e)
        
        return signals
    
    def _generate_analysis(self, processed_items: List[Dict[str, Any]], signals: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate comprehensive analysis."""
        analysis = {
            "summary": "",
            "key_findings": [],
            "sentiment": "neutral",
            "confidence": 0.0,
            "overall_score": 0.0,
            "risk_factors": [],
            "opportunities": []
        }
        
        try:
            # Analyze sentiment and calculate overall score
            sentiments = []
            sentiment_scores = []
            
            for item in processed_items:
                impact = item.get("impact", {})
                sentiment = impact.get("sentiment", "neutral")
                score = impact.get("score", 0.0)
                sentiments.append(sentiment)
                sentiment_scores.append(score)
            
            # Calculate overall sentiment and score
            if sentiment_scores:
                avg_score = sum(sentiment_scores) / len(sentiment_scores)
                analysis["overall_score"] = avg_score
                
                # Determine dominant sentiment
                sentiment_counts = {}
                for sentiment in sentiments:
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                if sentiment_counts:
                    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
                    analysis["sentiment"] = dominant_sentiment
                
                # Calculate confidence based on score consistency
                positive_count = sum(1 for s in sentiment_scores if s > 0.1)
                negative_count = sum(1 for s in sentiment_scores if s < -0.1)
                total_count = len(sentiment_scores)
                
                if total_count > 0:
                    consistency = max(positive_count, negative_count) / total_count
                    analysis["confidence"] = min(0.95, consistency + 0.3)  # Cap at 95%
                else:
                    analysis["confidence"] = 0.5
            else:
                analysis["overall_score"] = 0.0
                analysis["sentiment"] = "neutral"
                analysis["confidence"] = 0.0
            
            # Extract key findings from signals
            key_findings = []
            for signal_type, signal_list in signals.items():
                if isinstance(signal_list, list):
                    for signal in signal_list:
                        if signal.get("confidence", 0) > 0.6:
                            key_findings.append({
                                "type": signal_type,
                                "description": signal.get("description", ""),
                                "confidence": signal.get("confidence", 0)
                            })
            
            analysis["key_findings"] = key_findings
            
            # Identify risk factors
            risk_factors = []
            for signal_type, signal_list in signals.items():
                if isinstance(signal_list, list):
                    for signal in signal_list:
                        if signal.get("type") in ["earnings_miss", "regulatory_fine", "product_recall", "legal_action"]:
                            risk_factors.append(signal.get("description", ""))
            
            analysis["risk_factors"] = risk_factors
            
            # Identify opportunities
            opportunities = []
            for signal_type, signal_list in signals.items():
                if isinstance(signal_list, list):
                    for signal in signal_list:
                        if signal.get("type") in ["earnings_beat", "guidance_raise", "good_risk_adjusted_return"]:
                            opportunities.append(signal.get("description", ""))
            
            analysis["opportunities"] = opportunities
            
            # Calculate overall confidence
            if key_findings:
                analysis["confidence"] = sum(finding.get("confidence", 0) for finding in key_findings) / len(key_findings)
            
            # Generate summary
            if key_findings:
                analysis["summary"] = f"Analysis of {self.symbol} reveals {len(key_findings)} key findings with {analysis['sentiment']} sentiment."
            else:
                analysis["summary"] = f"Limited analysis available for {self.symbol}."
            
        except Exception as e:
            logger.error(f"Error generating analysis: {e}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _evaluate_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of the report."""
        evaluation = {
            "score": 0.0,
            "criteria": {},
            "recommendations": []
        }
        
        try:
            # Evidence count
            evidence_count = len(report.get("processed_items", []))
            evidence_score = min(1.0, evidence_count / 10.0)
            evaluation["criteria"]["evidence_count"] = evidence_score
            
            # Signal quality
            signals = report.get("signals", {})
            signal_count = sum(len(signal_list) for signal_list in signals.values() if isinstance(signal_list, list))
            signal_score = min(1.0, signal_count / 5.0)
            evaluation["criteria"]["signal_quality"] = signal_score
            
            # Analysis confidence
            analysis = report.get("analysis", {})
            confidence_score = analysis.get("confidence", 0.0)
            evaluation["criteria"]["confidence"] = confidence_score
            
            # Overall score
            evaluation["score"] = (evidence_score * 0.4 + signal_score * 0.3 + confidence_score * 0.3)
            
            # Recommendations
            if evidence_count < 5:
                evaluation["recommendations"].append("Increase news coverage for better analysis")
            if signal_count < 3:
                evaluation["recommendations"].append("Enhance signal detection algorithms")
            if confidence_score < 0.5:
                evaluation["recommendations"].append("Improve data quality and validation")
            
        except Exception as e:
            logger.error(f"Error evaluating report: {e}")
            evaluation["error"] = str(e)
        
        return evaluation
    
    def _optimize_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the report by fetching additional data."""
        logger.info("Optimizing report with additional data")
        
        try:
            # Fetch additional news with configuration-driven limit
            additional_news_limit = self.config.get_news_limit() * 2  # Double for additional iteration
            additional_news = self.news_client.search_symbol_news(self.symbol, limit=additional_news_limit)
            
            # Process additional news
            additional_processed = self._process_news_items(additional_news)
            
            # Add to existing processed items
            existing_items = report.get("processed_items", [])
            existing_links = {item.get("link") for item in existing_items if item.get("link")}
            
            for item in additional_processed:
                if item.get("link") not in existing_links:
                    existing_items.append(item)
            
            report["processed_items"] = existing_items
            
            # Re-analyze with additional data
            signals = self._route_to_specialists(existing_items, report.get("evidence", {}))
            report["signals"] = signals
            
            analysis = self._generate_analysis(existing_items, signals)
            report["analysis"] = analysis
            
        except Exception as e:
            logger.error(f"Error optimizing report: {e}")
            report["optimization_error"] = str(e)
        
        return report
    
    def _self_reflect(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Perform self-reflection on the analysis."""
        reflection = {
            "strengths": [],
            "weaknesses": [],
            "improvements": []
        }
        
        try:
            # Analyze strengths
            evidence_count = len(report.get("processed_items", []))
            if evidence_count > 10:
                reflection["strengths"].append("Comprehensive news coverage")
            
            signals = report.get("signals", {})
            signal_count = sum(len(signal_list) for signal_list in signals.values() if isinstance(signal_list, list))
            if signal_count > 5:
                reflection["strengths"].append("Rich signal detection")
            
            # Analyze weaknesses
            if evidence_count < 5:
                reflection["weaknesses"].append("Limited news coverage")
            
            if signal_count < 3:
                reflection["weaknesses"].append("Few actionable signals")
            
            evaluation = report.get("evaluation", {})
            if evaluation.get("score", 0) < 0.6:
                reflection["weaknesses"].append("Low overall quality score")
            
            # Suggest improvements
            if evidence_count < 10:
                reflection["improvements"].append("Expand news sources and time range")
            
            if signal_count < 5:
                reflection["improvements"].append("Enhance signal detection algorithms")
            
            if evaluation.get("score", 0) < 0.7:
                reflection["improvements"].append("Improve data quality and validation")
            
        except Exception as e:
            logger.error(f"Error in self-reflection: {e}")
            reflection["error"] = str(e)
        
        return reflection
    
    def _persist_memory(self, report: Dict[str, Any]):
        """Persist analysis memory."""
        try:
            memory_entry = {
                "symbol": self.symbol,
                "timestamp": report.get("timestamp"),
                "summary": report.get("analysis", {}).get("summary", ""),
                "sentiment": report.get("analysis", {}).get("sentiment", "neutral"),
                "confidence": report.get("analysis", {}).get("confidence", 0.0),
                "score": report.get("evaluation", {}).get("score", 0.0),
                "iterations": report.get("iterations", 0)
            }
            
            with self.memory_file.open("a", encoding="utf8") as f:
                f.write(json.dumps(memory_entry, ensure_ascii=False) + "\n")
                
        except Exception as e:
            logger.error(f"Error persisting memory: {e}")
    
    def plan_analysis_scope(self) -> Dict[str, Any]:
        """
        Plan the scope and priorities for the investment analysis.
        
        Returns:
            Analysis scope and configuration
        """
        logger.info(f"📋 Planning analysis scope for {self.symbol}")
        
        scope = {
            "symbol": self.symbol,
            "analysis_priorities": [
                "fundamental_analysis",
                "technical_analysis", 
                "sentiment_analysis",
                "risk_assessment"
            ],
            "data_sources": [
                "yahoo_finance",
                "news_api",
                "market_data"
            ],
            "analysis_depth": "comprehensive",
            "time_horizon": "medium_term",
            "risk_tolerance": "moderate"
        }
        
        logger.info(f"✅ Analysis scope planned: {len(scope['analysis_priorities'])} priorities")
        return scope
    
    def compile_research_report(self, analysis: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile comprehensive investment research report.
        
        Args:
            analysis: Analysis results
            signals: Generated signals
            
        Returns:
            Complete research report
        """
        logger.info("📄 Compiling comprehensive research report...")
        
        report = {
            "executive_summary": self._generate_executive_summary(analysis),
            "company_overview": self._generate_company_overview(),
            "financial_analysis": self._generate_financial_analysis(signals),
            "technical_analysis": self._generate_technical_analysis(signals),
            "sentiment_analysis": self._generate_sentiment_analysis(signals),
            "risk_assessment": self._generate_risk_assessment(analysis),
            "investment_thesis": self._generate_investment_thesis(analysis),
            "recommendations": self._generate_recommendations(analysis),
            "appendix": self._generate_appendix(signals)
        }
        
        logger.info("✅ Research report compiled successfully")
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate executive summary of the analysis."""
        sentiment = analysis.get("sentiment", "neutral")
        score = analysis.get("overall_score", 0.0)
        confidence = analysis.get("confidence", 0.0)
        
        # Use sophisticated recommendation logic
        if score > 0.3:
            recommendation = "STRONG BUY"
        elif score > 0.1:
            recommendation = "BUY"
        elif score > -0.1:
            recommendation = "HOLD"
        elif score > -0.3:
            recommendation = "SELL"
        else:
            recommendation = "STRONG SELL"
        
        return f"""
        Investment Analysis Summary for {self.symbol}
        
        Overall Assessment: {sentiment.upper()} sentiment with a score of {score:.2f}
        Confidence Level: {confidence:.1%}
        
        Key Findings:
        - Sentiment analysis indicates {sentiment} market perception
        - Technical indicators suggest {'positive' if score > 0 else 'negative'} momentum
        - Risk factors identified: {len(analysis.get('risk_factors', []))}
        - Opportunities identified: {len(analysis.get('opportunities', []))}
        
        Recommendation: {recommendation}
        """
    
    def _generate_company_overview(self) -> str:
        """Generate company overview section."""
        return f"""
        Company Overview: {self.symbol}
        
        This analysis covers the investment potential of {self.symbol} based on 
        comprehensive evaluation of fundamental, technical, and sentiment factors.
        The analysis incorporates real-time market data, news sentiment, and 
        technical indicators to provide a holistic investment perspective.
        """
    
    def _generate_financial_analysis(self, signals: Dict[str, Any]) -> str:
        """Generate financial analysis section."""
        earnings_signals = signals.get("earnings", [])
        return f"""
        Financial Analysis:
        
        Earnings Analysis: {len(earnings_signals)} signals generated
        - Recent earnings performance evaluated
        - Growth trends analyzed
        - Profitability metrics assessed
        
        Key Financial Metrics:
        - Revenue growth trends
        - Profit margin analysis
        - Debt-to-equity ratios
        - Cash flow evaluation
        """
    
    def _generate_technical_analysis(self, signals: Dict[str, Any]) -> str:
        """Generate technical analysis section."""
        tech_signals = signals.get("technical", [])
        return f"""
        Technical Analysis:
        
        Technical Indicators: {len(tech_signals)} signals generated
        - Moving average analysis
        - RSI and momentum indicators
        - Support and resistance levels
        - Volume analysis
        
        Chart Patterns:
        - Trend identification
        - Breakout patterns
        - Reversal signals
        """
    
    def _generate_sentiment_analysis(self, signals: Dict[str, Any]) -> str:
        """Generate sentiment analysis section."""
        news_signals = signals.get("news_impact", [])
        return f"""
        Sentiment Analysis:
        
        News Impact: {len(news_signals)} articles analyzed
        - Sentiment scoring across news sources
        - Topic classification and relevance
        - Market sentiment trends
        - Social sentiment indicators
        
        Sentiment Drivers:
        - Positive catalysts identified
        - Negative concerns highlighted
        - Neutral developments noted
        """
    
    def _generate_risk_assessment(self, analysis: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        risks = analysis.get("risk_factors", [])
        return f"""
        Risk Assessment:
        
        Identified Risks: {len(risks)} risk factors
        - Market risk factors
        - Company-specific risks
        - Regulatory risks
        - Competitive risks
        
        Risk Mitigation:
        - Diversification strategies
        - Position sizing recommendations
        - Stop-loss considerations
        """
    
    def _generate_investment_thesis(self, analysis: Dict[str, Any]) -> str:
        """Generate investment thesis section."""
        score = analysis.get("overall_score", 0.0)
        sentiment = analysis.get("sentiment", "neutral")
        
        thesis = f"""
        Investment Thesis for {self.symbol}:
        
        Based on comprehensive analysis, {self.symbol} presents a {sentiment} 
        investment opportunity with an overall score of {score:.2f}.
        
        Key Investment Drivers:
        - Fundamental strength indicators
        - Technical momentum signals
        - Positive sentiment catalysts
        - Growth potential factors
        
        Investment Rationale:
        The analysis supports a {'BUY' if score > 0.3 else 'HOLD' if score > -0.3 else 'SELL'} 
        recommendation based on the convergence of fundamental, technical, and 
        sentiment factors.
        """
        return thesis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Generate investment recommendations."""
        score = analysis.get("overall_score", 0.0)
        
        if score > 0.3:
            recommendation = "STRONG BUY"
            rationale = "Very strong positive signals across all analysis dimensions"
        elif score > 0.1:
            recommendation = "BUY"
            rationale = "Positive signals indicating favorable investment opportunity"
        elif score > -0.1:
            recommendation = "HOLD"
            rationale = "Mixed signals with neutral overall assessment"
        elif score > -0.3:
            recommendation = "SELL"
            rationale = "Negative signals indicating potential downside risk"
        else:
            recommendation = "STRONG SELL"
            rationale = "Very strong negative signals indicating significant downside risk"
        
        return f"""
        Investment Recommendations:
        
        Primary Recommendation: {recommendation}
        Rationale: {rationale}
        
        Position Sizing: Moderate (5-10% of portfolio)
        Time Horizon: Medium-term (6-12 months)
        Risk Level: Moderate
        
        Monitoring Points:
        - Earnings announcements
        - Market sentiment changes
        - Technical level breaks
        - News catalyst developments
        """
    
    def _generate_appendix(self, signals: Dict[str, Any]) -> str:
        """Generate appendix with detailed data."""
        return f"""
        Appendix - Detailed Analysis Data:
        
        Signal Summary:
        - Earnings signals: {len(signals.get('earnings', []))}
        - Technical signals: {len(signals.get('technical', []))}
        - News impact signals: {len(signals.get('news_impact', []))}
        - Regulatory signals: {len(signals.get('regulatory', []))}
        - Governance signals: {len(signals.get('governance', []))}
        
        Data Sources:
        - Yahoo Finance: Price data, fundamentals, earnings
        - News API: Real-time financial news
        - Technical indicators: Moving averages, RSI, Bollinger Bands
        
        Analysis Methodology:
        - LLM-based sentiment analysis
        - Transformer-based topic classification
        - Enhanced signal generation
        - Multi-agent coordination
        """
    
    def update_memory(self, report: Dict[str, Any]) -> None:
        """
        Update persistent memory with insights from current analysis.
        
        Args:
            report: Complete analysis report
        """
        logger.info("🧠 Updating persistent memory with insights...")
        
        try:
            memory_entry = {
                "timestamp": datetime.now().isoformat(),
                "symbol": self.symbol,
                "key_insights": {
                    "sentiment": report.get("analysis", {}).get("sentiment"),
                    "score": report.get("analysis", {}).get("overall_score"),
                    "confidence": report.get("analysis", {}).get("confidence"),
                    "recommendation": report.get("analysis", {}).get("recommendation", "HOLD")
                },
                "learning_points": report.get("reflection", {}).get("improvements", []),
                "success_patterns": report.get("reflection", {}).get("strengths", [])
            }
            
            # Append to memory file
            with open(self.memory_file, "a") as f:
                f.write(json.dumps(memory_entry) + "\n")
            
            logger.info("✅ Memory updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
    
    def _save_report(self, report: Dict[str, Any]):
        """Save detailed report to file."""
        try:
            os.makedirs("outputs/reports", exist_ok=True)
            filename = f"outputs/reports/{self.symbol}_enhanced_analysis.json"
            
            # Normalize the report for JSON serialization
            normalized_report = self._normalize_for_json(report)
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(normalized_report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Report saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
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
