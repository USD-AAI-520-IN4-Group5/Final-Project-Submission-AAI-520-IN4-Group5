"""
Streamlit Application for Financial Agentic AI System
A web interface for the investment research agent system.
"""
import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import our agents and workflows
from src.agents.enhanced_investment_agent import EnhancedInvestmentAgent
from src.tools.news_api import NewsAPIClient
from src.tools.yfinance_client import YFinanceClient
from src.utils.text_processing import extract_keywords, simple_sentiment_score, classify_topic
from src.utils.config_loader import ConfigLoader

# Page configuration
st.set_page_config(
    page_title="Financial Agentic AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .news-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_gradient_header(title, icon="📊", subtitle=None):
    """Create a consistent gradient header for sections."""
    subtitle_html = f'<p style="margin: 0; font-size: 14px; color: rgba(255,255,255,0.8);">{subtitle}</p>' if subtitle else ""
    
    return f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
            <span style='font-size: 20px; margin-right: 12px;'>{icon}</span>
            <h3 style='margin: 0; color: white; font-weight: 600; font-size: 22px;'>{title}</h3>
        </div>
        {subtitle_html}
    </div>
    """

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Enhanced Investment Research Agent</h1>', unsafe_allow_html=True)
    
    # Add workflow phase indicator
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <div style='display: flex; align-items: center; margin-bottom: 10px;'>
            <span style='font-size: 20px; margin-right: 12px;'>🔄</span>
            <h3 style='margin: 0; color: white; font-weight: 600; font-size: 22px;'>Autonomous Analysis Workflow</h3>
        </div>
        <p style='margin: 0; font-size: 16px; color: rgba(255,255,255,0.9); line-height: 1.5;'>
            The agent follows a structured 5-phase workflow: <strong>Planning</strong> → <strong>Execution</strong> → <strong>Synthesis</strong> → <strong>Evaluation</strong> → <strong>Self-Reflection</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Symbol input
        default_symbol = st.session_state.get('suggested_symbol', st.session_state.get('current_symbol', 'AAPL'))
        symbol = st.text_input(
            "Stock Symbol",
            value=default_symbol,
            help="Enter a stock symbol to analyze (e.g., AAPL, MSFT, GOOGL)"
        ).upper()
        
        # Store the current symbol in session state
        if symbol:
            st.session_state.current_symbol = symbol
        
        # Clear suggested symbol after using it (but keep current_symbol)
        if 'suggested_symbol' in st.session_state:
            del st.session_state.suggested_symbol
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        max_iterations = st.slider("Max Iterations", 1, 5, 2)
        news_limit = st.slider("News Articles Limit", 5, 50, 10)
        
        # Finance filtering configuration
        st.subheader("🔍 Finance News Filtering")
        enable_finance_filter = st.checkbox(
            "Enable Finance-Specific Filtering", 
            value=True,
            help="Filter news to only include finance-related articles"
        )
        
        if enable_finance_filter:
            min_relevance = st.slider(
                "Minimum Finance Relevance Score", 
                0.0, 1.0, 0.3, 0.1,
                help="Higher values = more strict filtering"
            )
        else:
            min_relevance = 0.0
        
        # API Configuration
        st.subheader("API Configuration")
        api_key = st.text_input(
            "News API Key",
            value="",
            type="password",
            help="Your News API key"
        )
        
        # Load default API keys from config
        config_loader = ConfigLoader()
        default_openai_key = config_loader.get_openai_api_key()
        default_huggingface_token = config_loader.get_huggingface_token()
        
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)",
            value=default_openai_key,
            type="password",
            help="For advanced LLM-based analysis. Leave empty to use local transformer models."
        )
        
        huggingface_token = st.text_input(
            "Hugging Face Token (Optional)",
            value=default_huggingface_token,
            type="password",
            help="For accessing Hugging Face transformer models. Leave empty to use public models."
        )
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary"):
            # Validate required API key
            if not api_key:
                st.error("❌ News API Key is required to fetch financial news. Please enter your API key.")
                st.stop()
            
            # Create configuration with user settings
            config_data = {
                'news_filtering': {
                    'enable_finance_filter': enable_finance_filter,
                    'min_finance_relevance': min_relevance,
                    'llm_based_filtering': True,
                    'exclude_keywords': [
                        'sports', 'entertainment', 'celebrity', 'movie', 'music',
                        'gaming', 'weather', 'politics', 'election', 'crime',
                        'accident', 'health', 'medical'
                    ],
                    'preferred_sources': [
                        'bloomberg', 'reuters', 'wsj', 'financial times', 'cnbc',
                        'marketwatch', 'yahoo finance', 'seeking alpha', 'benzinga'
                    ]
                },
                'analysis': {
                    'max_iterations': max_iterations,
                    'news_limit': news_limit,
                    'confidence_threshold': 0.5
                },
                'api': {
                    'news_api_key': api_key,
                    'openai_api_key': openai_api_key,
                    'huggingface_token': huggingface_token
                }
            }
            
            run_analysis(symbol, max_iterations, news_limit, api_key, config_data, openai_api_key, huggingface_token)
    
    # Handle improvement triggers
    if st.session_state.get('trigger_improvement', False):
        st.markdown("---")
        st.markdown("### 🔄 Running Improved Analysis")
        
        # Get improvement parameters
        improvement_focus = st.session_state.get('improvement_focus', 'all')
        selected_improvements = st.session_state.get('selected_improvements', [])
        
        # Create configuration with user settings
        config_data = {
            'news_filtering': {
                'enable_finance_filter': enable_finance_filter,
                'min_finance_relevance': min_relevance,
                'llm_based_filtering': True,
                'exclude_keywords': [
                    'sports', 'entertainment', 'celebrity', 'movie', 'music',
                    'gaming', 'weather', 'politics', 'election', 'crime',
                    'accident', 'health', 'medical'
                ],
                'preferred_sources': [
                    'bloomberg', 'reuters', 'wsj', 'financial times', 'cnbc',
                    'marketwatch', 'yahoo finance', 'seeking alpha', 'benzinga'
                ]
            },
            'analysis': {
                'max_iterations': max_iterations,
                'news_limit': news_limit,
                'confidence_threshold': 0.5
            },
            'api': {
                'news_api_key': api_key,
                'openai_api_key': openai_api_key,
                'huggingface_token': huggingface_token
            }
        }
        
        # Run improved analysis
        run_analysis(symbol, max_iterations, news_limit, api_key, config_data, openai_api_key, huggingface_token, improvement_focus, selected_improvements)
        
        # Clear improvement trigger
        st.session_state.trigger_improvement = False
        st.session_state.improvement_focus = None
        st.session_state.selected_improvements = None
    
    # Main content area
    if 'analysis_results' in st.session_state:
        display_results()
    else:
        display_welcome()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6c757d;'>"
        "Enhanced Investment Research Agent • Built with Streamlit • Powered by News API"
        "</div>",
        unsafe_allow_html=True
    )

def run_analysis(symbol, max_iterations, news_limit, api_key, config_data=None, openai_api_key=None, huggingface_token=None, improvement_focus=None, selected_improvements=None):
    """Run the financial analysis for the given symbol with optional improvements."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Initialize agent with configuration
        status_text.text("🔧 Initializing Enhanced Investment Research Agent...")
        progress_bar.progress(10)
        
        # Create configuration loader with user settings
        if config_data:
            # Add user-provided API keys to config_data
            if 'api' not in config_data:
                config_data['api'] = {}
            
            # Override with user-provided API keys
            if api_key:
                config_data['api']['news_api_key'] = api_key
            if openai_api_key:
                config_data['api']['openai_api_key'] = openai_api_key
            if huggingface_token:
                config_data['api']['huggingface_token'] = huggingface_token
            
            # Create a temporary config file with user settings
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                temp_config_path = f.name
            
            config = ConfigLoader(temp_config_path)
        else:
            # Create config with user-provided API keys
            config_data = {
                'api': {
                    'news_api_key': api_key or "",
                    'openai_api_key': openai_api_key or "",
                    'huggingface_token': huggingface_token or ""
                }
            }
            
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_data, f)
                temp_config_path = f.name
            
            config = ConfigLoader(temp_config_path)
        
        agent = EnhancedInvestmentAgent(symbol=symbol, max_iterations=max_iterations, config=config, openai_api_key=openai_api_key, improvement_focus=improvement_focus, selected_improvements=selected_improvements)
        
        # Step 2: Fetch data
        status_text.text("📊 Fetching stock data and news...")
        progress_bar.progress(30)
        
        # Store API key in session state for the agent
        st.session_state.api_key = api_key
        
        # Step 3: Run enhanced analysis
        status_text.text("🧠 Running enhanced AI analysis...")
        progress_bar.progress(60)
        
        # Run the enhanced agent analysis
        results = agent.act()
        
        # Step 4: Process results
        status_text.text("📊 Processing results...")
        progress_bar.progress(90)
        
        # Store results in session state
        st.session_state.analysis_results = results
        st.session_state.symbol = symbol
        st.session_state.analysis_time = datetime.now()
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis completed!")
        
        # Clear progress indicators after a short delay
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Clean up temporary config file if created
        if config_data and 'temp_config_path' in locals():
            import os
            try:
                os.unlink(temp_config_path)
            except:
                pass
        
        st.success(f"Analysis completed for {symbol}!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        
        # Clean up temporary config file if created
        if config_data and 'temp_config_path' in locals():
            import os
            try:
                os.unlink(temp_config_path)
            except:
                pass

def display_welcome():
    """Display welcome message with visual elements."""
    
    # Main welcome card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h2 style="margin: 0 0 15px 0; text-align: center;">🤖 Enhanced Investment Research Agent</h2>
        <p style="margin: 0; text-align: center; font-size: 18px; opacity: 0.9;">
            Autonomous financial analysis powered by AI agents and advanced visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #28a745;">🎯 Autonomous Analysis</h4>
            <ul style="margin: 0; padding-left: 20px; color: #333;">
                <li>5-phase structured workflow</li>
                <li>Dynamic data retrieval</li>
                <li>Intelligent processing</li>
                <li>Self-reflection & learning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 5px solid #2196f3; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #2196f3;">📊 Visual Analytics</h4>
            <ul style="margin: 0; padding-left: 20px; color: #333;">
                <li>Interactive charts & graphs</li>
                <li>Real-time sentiment analysis</li>
                <li>Financial metrics dashboard</li>
                <li>Workflow progress tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800; margin-bottom: 20px;">
            <h4 style="margin: 0 0 10px 0; color: #ff9800;">🚀 Advanced Features</h4>
            <ul style="margin: 0; padding-left: 20px; color: #333;">
                <li>Multi-agent processing</li>
                <li>LLM-based analysis</li>
                <li>Comprehensive reporting</li>
                <li>Persistent memory</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
            <h5 style="margin: 0 0 15px 0; color: #333;">📝 How to Use</h5>
            <ol style="margin: 0; padding-left: 20px; color: #333;">
                <li><strong>Enter Stock Symbol:</strong> Input any ticker (AAPL, MSFT, etc.)</li>
                <li><strong>Configure Settings:</strong> Adjust analysis parameters</li>
                <li><strong>Provide API Keys:</strong> News API required</li>
                <li><strong>Run Analysis:</strong> Agent executes autonomously</li>
                <li><strong>Explore Results:</strong> Interactive visualizations</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
            <h5 style="margin: 0 0 15px 0; color: #333;">📊 Dashboard Features</h5>
            <ul style="margin: 0; padding-left: 20px; color: #333;">
                <li><strong>Sentiment Dashboard:</strong> Interactive sentiment analysis</li>
                <li><strong>Financial Metrics:</strong> Price charts & indicators</li>
                <li><strong>News Analysis:</strong> Publisher & topic breakdown</li>
                <li><strong>Workflow Progress:</strong> Visual phase tracking</li>
                <li><strong>Research Report:</strong> Comprehensive analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Popular symbols with visual buttons
    st.markdown("### 💡 Popular Symbols to Try")
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    cols = st.columns(5)
    
    for i, symbol in enumerate(symbols):
        with cols[i]:
            if st.button(f"📈 {symbol}", key=f"symbol_{symbol}", use_container_width=True):
                st.session_state.suggested_symbol = symbol
                st.session_state.current_symbol = symbol
                st.rerun()
    
    # Additional info
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 0; text-align: center; color: #1976d2;">
            <strong>💡 Tip:</strong> The agent learns from each analysis and stores insights in persistent memory for continuous improvement.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_results():
    """Display analysis results with visualization-first approach."""
    
    results = st.session_state.analysis_results
    symbol = st.session_state.symbol
    processed_items = results.get('processed_items', [])
    analysis = results.get('analysis', {})
    
    # Header with symbol and timestamp
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <div style='display: flex; align-items: center; margin-bottom: 8px;'>
            <span style='font-size: 20px; margin-right: 12px;'>📈</span>
            <h2 style='margin: 0; color: white; font-weight: 700; font-size: 28px;'>Analysis Results for {symbol}</h2>
        </div>
        <p style='margin: 0; font-size: 14px; color: rgba(255,255,255,0.8);'>
            Analysis completed at: {st.session_state.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Dashboard - Visual Overview
    display_main_dashboard(results, symbol)
    
    # Tabs for different views - Reorganized for better UX
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Sentiment Dashboard", 
        "💰 Financial Metrics",
        "📰 News Analysis", 
        "🔄 Workflow Progress",
        "🔍 Raw Data"
    ])
    
    with tab1:
        display_sentiment_dashboard(processed_items, analysis)
    
    with tab2:
        display_financial_metrics(results)
    
    with tab3:
        display_news_analysis_visual(processed_items)
    
    with tab4:
        display_workflow_progress(results)

    with tab5:
        display_raw_data(results)

def display_main_dashboard(results, symbol):
    """Display main visual dashboard with key metrics and charts."""
    
    # Extract key data
    analysis = results.get('analysis', {})
    evaluation = results.get('evaluation', {})
    processed_items = results.get('processed_items', [])
    signals = results.get('signals', {})
    
    # Main KPI Row
    st.markdown("### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        overall_score = analysis.get('overall_score', 0)
        sentiment = analysis.get('sentiment', 'neutral')
        sentiment_color = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(sentiment, '⚪')
        st.metric("Overall Sentiment", f"{sentiment_color} {sentiment.title()}", f"{overall_score:.2f}")
    
    with col2:
        confidence = analysis.get('confidence', 0)
        st.metric("Analysis Confidence", f"{confidence:.1%}")
    
    with col3:
        eval_score = evaluation.get('score', 0)
        st.metric("Quality Score", f"{eval_score:.1%}")
    
    with col4:
        news_count = len(processed_items)
        st.metric("News Articles", news_count)
    
    with col5:
        total_signals = sum(len(signal_list) if isinstance(signal_list, list) else 0 
                          for signal_list in signals.values())
        st.metric("Signals Detected", total_signals)
    
    # Recommendation Card
    st.markdown("### 💡 Investment Recommendation")
    
    # Determine recommendation based on score
    score = analysis.get('overall_score', 0)
    if score > 0.3:
        recommendation = "STRONG BUY"
        rec_color = "#28a745"
        rec_icon = "🚀"
    elif score > 0.1:
        recommendation = "BUY"
        rec_color = "#20c997"
        rec_icon = "📈"
    elif score > -0.1:
        recommendation = "HOLD"
        rec_color = "#ffc107"
        rec_icon = "⏸️"
    elif score > -0.3:
        recommendation = "SELL"
        rec_color = "#fd7e14"
        rec_icon = "📉"
    else:
        recommendation = "STRONG SELL"
        rec_color = "#dc3545"
        rec_icon = "⚠️"
    
    st.markdown(f"""
    <div style="background-color: {rec_color}20; padding: 20px; border-radius: 10px; border-left: 5px solid {rec_color}; margin: 10px 0;">
        <h2 style="color: {rec_color}; margin: 0; display: flex; align-items: center;">
            {rec_icon} {recommendation}
        </h2>
        <p style="margin: 5px 0 0 0; color: #666;">
            Based on comprehensive analysis with {confidence:.1%} confidence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Sentiment Overview Chart
    if processed_items:
        st.markdown("### 📊 Quick Sentiment Overview")
        
        # Prepare sentiment data
        sentiment_data = []
        for item in processed_items:
            impact = item.get('impact', {})
            sentiment_data.append({
                'Sentiment': impact.get('sentiment', 'neutral'),
                'Score': impact.get('score', 0),
                'Title': item.get('title', 'Untitled')[:30] + '...'
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = sentiment_df['Sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                }
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Sentiment score histogram
            fig_hist = px.histogram(
                sentiment_df,
                x='Score',
                nbins=15,
                title="Sentiment Score Distribution",
                color='Sentiment',
                color_discrete_map={
                    'positive': '#28a745',
                    'negative': '#dc3545', 
                    'neutral': '#6c757d'
                }
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

def display_sentiment_dashboard(processed_items, analysis):
    """Display comprehensive sentiment analysis dashboard."""
    
    if not processed_items:
        st.warning("No sentiment data available.")
        return
    
    st.markdown(create_gradient_header("Comprehensive Sentiment Analysis Dashboard", "🎭", "Detailed sentiment analysis across all news articles"), unsafe_allow_html=True)
    
    # Prepare comprehensive sentiment data
    sentiment_data = []
    for i, item in enumerate(processed_items):
        impact = item.get('impact', {})
        sentiment_data.append({
            'Article': f"Article {i+1}",
            'Title': item.get('title', 'Untitled'),
            'Short_Title': item.get('title', 'Untitled')[:40] + '...' if len(item.get('title', '')) > 40 else item.get('title', 'Untitled'),
            'Sentiment': impact.get('sentiment', 'neutral'),
            'Score': impact.get('score', 0),
            'Topic': item.get('topic', 'Unknown'),
            'Publisher': item.get('publisher', 'Unknown'),
            'Keywords': ', '.join(item.get('keywords', [])[:3])  # First 3 keywords
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Top row: Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sentiment_df['Score'].mean()
        st.metric("Average Sentiment Score", f"{avg_score:.3f}")
    
    with col2:
        positive_pct = len(sentiment_df[sentiment_df['Sentiment'] == 'positive']) / len(sentiment_df) * 100
        st.metric("Positive Articles", f"{positive_pct:.1f}%")
    
    with col3:
        negative_pct = len(sentiment_df[sentiment_df['Sentiment'] == 'negative']) / len(sentiment_df) * 100
        st.metric("Negative Articles", f"{negative_pct:.1f}%")
    
    with col4:
        neutral_pct = len(sentiment_df[sentiment_df['Sentiment'] == 'neutral']) / len(sentiment_df) * 100
        st.metric("Neutral Articles", f"{neutral_pct:.1f}%")
    
    # Interactive scatter plot
    st.markdown("### 🎯 Interactive Sentiment Analysis")
    
    # Add size column for better visualization
    sentiment_df['Size'] = abs(sentiment_df['Score']) * 20 + 5  # Scale for visibility
    
    fig_scatter = px.scatter(
        sentiment_df,
        x='Score',
        y='Topic',
        color='Sentiment',
        size='Size',
        hover_data=['Short_Title', 'Publisher', 'Keywords'],
        title="Article Sentiment Scores by Topic",
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545', 
            'neutral': '#6c757d'
        }
    )
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral Line")
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Sentiment by topic analysis
    st.markdown("### 🏷️ Sentiment Analysis by Topic")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stacked bar chart
        topic_sentiment = pd.crosstab(sentiment_df['Topic'], sentiment_df['Sentiment'])
        fig_stacked = px.bar(
            topic_sentiment,
            title="Sentiment Distribution by Topic",
            color_discrete_map={
                'positive': '#28a745',
                'negative': '#dc3545', 
                'neutral': '#6c757d'
            }
        )
        fig_stacked.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_stacked, use_container_width=True)
    
    with col2:
        # Heatmap
        fig_heatmap = px.imshow(
            topic_sentiment.T,
            labels=dict(x="Topic", y="Sentiment", color="Count"),
            title="Sentiment-Topic Heatmap",
            color_continuous_scale="RdYlGn"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Detailed sentiment table
    st.markdown("### 📋 Detailed Sentiment Analysis")
    
    # Create a more detailed table
    display_df = sentiment_df[['Short_Title', 'Sentiment', 'Score', 'Topic', 'Publisher']].copy()
    display_df['Score'] = display_df['Score'].round(3)
    
    # Color code the sentiment column
    def color_sentiment(val):
        if val == 'positive':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'negative':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #e2e3e5; color: #383d41'
    
    styled_df = display_df.style.applymap(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_df, use_container_width=True, height=400)

def display_financial_metrics(results):
    """Display financial metrics and analysis."""
    
    st.markdown(create_gradient_header("Financial Metrics Dashboard", "💹", "Key financial indicators and performance metrics"), unsafe_allow_html=True)
    
    # Extract financial data
    evidence = results.get('evidence', {})
    price_data = evidence.get('price_data', {})
    signals = results.get('signals', {})
    
    if not price_data or not price_data.get('data'):
        st.warning("No financial data available.")
        return
    
    # Convert price data to DataFrame
    price_df = pd.DataFrame(price_data['data'])
    
    if price_df.empty:
        st.warning("Price data is empty.")
        return
    
    # Convert date column
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df = price_df.sort_values('Date')
    
    # Price chart
    st.markdown("### 📈 Price Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price trend
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=price_df['Date'],
            y=price_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Add moving averages if available
        if 'MA_20' in price_df.columns and not price_df['MA_20'].isna().all():
            fig_price.add_trace(go.Scatter(
                x=price_df['Date'],
                y=price_df['MA_20'],
                mode='lines',
                name='MA 20',
                line=dict(color='orange', width=1, dash='dash')
            ))
        
        fig_price.update_layout(
            title="Stock Price Trend",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=400
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Volume analysis
        fig_volume = go.Figure()
        fig_volume.add_trace(go.Bar(
            x=price_df['Date'],
            y=price_df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_volume.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume",
            height=400
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Financial metrics
    st.markdown("### 📊 Key Financial Metrics")
    
    # Calculate some basic metrics
    latest_price = price_df['Close'].iloc[-1]
    price_change = price_df['Close'].iloc[-1] - price_df['Close'].iloc[-2] if len(price_df) > 1 else 0
    price_change_pct = (price_change / price_df['Close'].iloc[-2]) * 100 if len(price_df) > 1 else 0
    
    avg_volume = price_df['Volume'].mean()
    latest_volume = price_df['Volume'].iloc[-1]
    volume_change = latest_volume - avg_volume
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Price", f"${latest_price:.2f}", f"{price_change:+.2f}")
    
    with col2:
        st.metric("Price Change", f"{price_change_pct:+.2f}%")
    
    with col3:
        st.metric("Avg Volume", f"{avg_volume:,.0f}")
    
    with col4:
        st.metric("Volume vs Avg", f"{volume_change:+,.0f}")
    
    # Technical indicators if available
    if any(col in price_df.columns for col in ['RSI', 'MA_20', 'MA_50']):
        st.markdown("### 🔧 Technical Indicators")
        
        tech_cols = [col for col in ['RSI', 'MA_20', 'MA_50'] if col in price_df.columns]
        
        if tech_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                # RSI if available
                if 'RSI' in price_df.columns and not price_df['RSI'].isna().all():
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=price_df['Date'],
                        y=price_df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple')
                    ))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                    fig_rsi.update_layout(title="RSI Indicator", height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                # Moving averages comparison
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(
                    x=price_df['Date'],
                    y=price_df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='black')
                ))
                
                if 'MA_20' in price_df.columns and not price_df['MA_20'].isna().all():
                    fig_ma.add_trace(go.Scatter(
                        x=price_df['Date'],
                        y=price_df['MA_20'],
                        mode='lines',
                        name='MA 20',
                        line=dict(color='blue')
                    ))
                
                if 'MA_50' in price_df.columns and not price_df['MA_50'].isna().all():
                    fig_ma.add_trace(go.Scatter(
                        x=price_df['Date'],
                        y=price_df['MA_50'],
                        mode='lines',
                        name='MA 50',
                        line=dict(color='red')
                    ))
                
                fig_ma.update_layout(title="Moving Averages", height=300)
                st.plotly_chart(fig_ma, use_container_width=True)

def display_news_analysis_visual(processed_items):
    """Display news analysis with enhanced visualizations."""
    
    if not processed_items:
        st.warning("No news analysis results available.")
        return
    
    st.markdown(create_gradient_header("News Analysis Dashboard", "📰", "Comprehensive news analysis and insights"), unsafe_allow_html=True)
    
    # News overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", len(processed_items))
    
    with col2:
        publishers = set(item.get('publisher', 'Unknown') for item in processed_items)
        st.metric("Unique Publishers", len(publishers))
    
    with col3:
        topics = set(item.get('topic', 'Unknown') for item in processed_items)
        st.metric("Topic Categories", len(topics))
    
    with col4:
        avg_keywords = sum(len(item.get('keywords', [])) for item in processed_items) / len(processed_items)
        st.metric("Avg Keywords", f"{avg_keywords:.1f}")
    
    # Publisher distribution
    st.markdown("### 📰 Publisher Analysis")
    
    publisher_data = [item.get('publisher', 'Unknown') for item in processed_items]
    publisher_counts = pd.Series(publisher_data).value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_publisher = px.pie(
            values=publisher_counts.values,
            names=publisher_counts.index,
            title="Articles by Publisher"
        )
        fig_publisher.update_layout(height=400)
        st.plotly_chart(fig_publisher, use_container_width=True)
    
    with col2:
        fig_publisher_bar = px.bar(
            x=publisher_counts.index,
            y=publisher_counts.values,
            title="Publisher Article Count"
        )
        fig_publisher_bar.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_publisher_bar, use_container_width=True)
    
    # Topic analysis
    st.markdown("### 🏷️ Topic Analysis")
    
    topic_data = [item.get('topic', 'Unknown') for item in processed_items]
    topic_counts = pd.Series(topic_data).value_counts()
    
    fig_topic = px.bar(
        x=topic_counts.values,
        y=topic_counts.index,
        orientation='h',
        title="Articles by Topic"
    )
    fig_topic.update_layout(height=400)
    st.plotly_chart(fig_topic, use_container_width=True)
    
    # Keyword analysis
    st.markdown("### 🔤 Keyword Analysis")
    
    all_keywords = []
    for item in processed_items:
        keywords = item.get('keywords', [])
        all_keywords.extend(keywords)
    
    if all_keywords:
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        
        fig_keywords = px.bar(
            x=keyword_counts.values,
            y=keyword_counts.index,
            orientation='h',
            title="Top 20 Keywords"
        )
        fig_keywords.update_layout(height=500)
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    # Interactive news table
    st.markdown("### 📋 Interactive News Table")
    
    # Prepare table data
    table_data = []
    for i, item in enumerate(processed_items):
        impact = item.get('impact', {})
        table_data.append({
            'Article': i + 1,
            'Title': item.get('title', 'Untitled')[:60] + '...' if len(item.get('title', '')) > 60 else item.get('title', 'Untitled'),
            'Publisher': item.get('publisher', 'Unknown'),
            'Topic': item.get('topic', 'Unknown'),
            'Sentiment': impact.get('sentiment', 'neutral'),
            'Score': round(impact.get('score', 0), 3),
            'Keywords': ', '.join(item.get('keywords', [])[:3])
        })
    
    news_df = pd.DataFrame(table_data)
    
    # Add color coding for sentiment
    def color_sentiment(val):
        if val == 'positive':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'negative':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #e2e3e5; color: #383d41'
    
    styled_news_df = news_df.style.applymap(color_sentiment, subset=['Sentiment'])
    st.dataframe(styled_news_df, use_container_width=True, height=400)

def display_workflow_progress(results):
    """Display workflow progress with visual indicators."""
    
    st.markdown(create_gradient_header("Autonomous Workflow Progress", "⚙️", "5-phase autonomous analysis workflow status"), unsafe_allow_html=True)
    
    # Workflow phases with visual progress
    phases = [
        ("📋 Planning", "Analysis scope definition and resource allocation"),
        ("📊 Execution", "Data retrieval and evidence gathering"),
        ("🧠 Synthesis", "Analysis generation and report compilation"),
        ("🔍 Evaluation", "Quality assessment and optimization"),
        ("🤔 Self-Reflection", "Performance analysis and memory integration")
    ]
    
    # Create progress visualization
    st.markdown("### 🎯 Workflow Completion Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    for i, (phase, description) in enumerate(phases):
        with [col1, col2, col3, col4, col5][i]:
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; border-radius: 10px; background-color: #e8f5e8; border: 2px solid #28a745;">
                <h4 style="margin: 0; color: #28a745;">{phase}</h4>
                <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">{description}</p>
                <div style="margin-top: 10px; font-size: 24px;">✅</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Analysis scope visualization
    analysis_scope = results.get("analysis_scope", {})
    if analysis_scope:
        st.markdown("### 📋 Analysis Scope")
        
        col1, col2 = st.columns(2)
        
        with col1:
            priorities = analysis_scope.get("analysis_priorities", [])
            if priorities:
                st.markdown("**Analysis Priorities:**")
                priority_df = pd.DataFrame({
                    'Priority': [p.replace('_', ' ').title() for p in priorities],
                    'Status': ['Active'] * len(priorities)
                })
                st.dataframe(priority_df, use_container_width=True)
        
        with col2:
            sources = analysis_scope.get("data_sources", [])
            if sources:
                st.markdown("**Data Sources:**")
                source_df = pd.DataFrame({
                    'Source': [s.replace('_', ' ').title() for s in sources],
                    'Status': ['Connected'] * len(sources)
                })
                st.dataframe(source_df, use_container_width=True)
    
    # Evidence gathering metrics
    evidence = results.get("evidence", {})
    if evidence:
        st.markdown("### 📊 Evidence Gathering Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            news_count = len(evidence.get("news", []))
            st.metric("News Articles", news_count)
        
        with col2:
            price_data = evidence.get("price_data", {})
            price_count = len(price_data.get("data", [])) if price_data else 0
            st.metric("Price Data Points", price_count)
        
        with col3:
            earnings_data = evidence.get("earnings", [])
            earnings_count = len(earnings_data) if earnings_data else 0
            st.metric("Earnings Records", earnings_count)
    
    # Signal generation visualization
    signals = results.get("signals", {})
    if signals:
        st.markdown("### 🎯 Signal Generation Analysis")
        
        signal_types = [
            ("News Impact", signals.get("news_impact", [])),
            ("Technical Analysis", signals.get("technical", [])),
            ("Earnings Analysis", signals.get("earnings", [])),
            ("Regulatory Analysis", signals.get("regulatory", [])),
            ("Governance Analysis", signals.get("governance", []))
        ]
        
        signal_data = []
        for signal_type, signal_list in signal_types:
            signal_data.append({
                'Signal Type': signal_type,
                'Count': len(signal_list) if signal_list else 0,
                'Status': 'Active' if signal_list else 'Inactive'
            })
        
        signal_df = pd.DataFrame(signal_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_signals = px.bar(
                signal_df,
                x='Signal Type',
                y='Count',
                title="Signals Generated by Type"
            )
            fig_signals.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_signals, use_container_width=True)
        
        with col2:
            st.dataframe(signal_df, use_container_width=True)
    
    # Evaluation results
    evaluation = results.get("evaluation", {})
    if evaluation:
        st.markdown("### 🔍 Evaluation Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = evaluation.get("score", 0)
            st.metric("Overall Score", f"{score:.1%}")
        
        with col2:
            completeness = evaluation.get("completeness", 0)
            st.metric("Completeness", f"{completeness:.1%}")
        
        with col3:
            consistency = evaluation.get("consistency", 0)
            st.metric("Consistency", f"{consistency:.1%}")
        
        # Quality indicators
        needs_optimization = evaluation.get("needs_optimization", False)
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: {'#f8d7da' if needs_optimization else '#d4edda'}; border-left: 4px solid {'#dc3545' if needs_optimization else '#28a745'}; color: {'#721c24' if needs_optimization else '#155724'};">
            <strong>Optimization Status:</strong> {'Needs Optimization' if needs_optimization else 'No Optimization Required'}
        </div>
        """, unsafe_allow_html=True)
    
    # Self-reflection insights
    reflection = results.get("reflection", {})
    if reflection:
        st.markdown("### 🤔 Self-Reflection Insights")
        
        strengths = reflection.get("strengths", [])
        improvements = reflection.get("improvements", [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if strengths:
                st.markdown("**Identified Strengths:**")
                strength_df = pd.DataFrame({
                    'Strength': strengths,
                    'Impact': ['High'] * len(strengths)
                })
                st.dataframe(strength_df, use_container_width=True)
        
        with col2:
            if improvements:
                st.markdown("**Areas for Improvement:**")
                improvement_df = pd.DataFrame({
                    'Improvement': improvements,
                    'Priority': ['High'] * len(improvements)
                })
                st.dataframe(improvement_df, use_container_width=True)
        
        # Improvement Trigger Section
        if improvements:
            st.markdown("---")
            st.markdown("### 🔄 Trigger Improved Analysis")
            
            st.markdown("""
            <div style="background-color: #e8f4fd; padding: 15px; border-radius: 10px; border-left: 4px solid #2196f3; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0; color: #1976d2;">💡 Continuous Improvement</h4>
                <p style="margin: 0; color: #333;">
                    The agent has identified areas for improvement. You can trigger a new analysis with enhanced parameters 
                    to address these specific issues and get better results.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Improved Analysis", key="improve_analysis", type="primary"):
                    st.session_state.trigger_improvement = True
                    st.session_state.improvement_focus = "all"
                    st.rerun()
            
            with col2:
                if st.button("📰 Expand News Sources", key="improve_news"):
                    st.session_state.trigger_improvement = True
                    st.session_state.improvement_focus = "news_sources"
                    st.rerun()
            
            with col3:
                if st.button("⏰ Extend Time Range", key="improve_time"):
                    st.session_state.trigger_improvement = True
                    st.session_state.improvement_focus = "time_range"
                    st.rerun()
            
            # Show improvement options
            st.markdown("#### 🎯 Improvement Options")
            
            improvement_options = st.multiselect(
                "Select specific improvements to apply:",
                options=improvements,
                default=improvements[:2] if len(improvements) >= 2 else improvements,
                key="improvement_selection"
            )
            
            if improvement_options:
                st.markdown("**Selected Improvements:**")
                for option in improvement_options:
                    st.markdown(f"• {option}")
                
                if st.button("🎯 Apply Selected Improvements", key="apply_selected", type="secondary"):
                    st.session_state.trigger_improvement = True
                    st.session_state.improvement_focus = "selected"
                    st.session_state.selected_improvements = improvement_options
                    st.rerun()

def display_workflow_details(results):
    """Display detailed autonomous workflow information."""
    st.markdown(create_gradient_header("Autonomous Workflow Details", "🔧", "Detailed workflow execution information"), unsafe_allow_html=True)
    
    # Analysis Scope
    analysis_scope = results.get("analysis_scope", {})
    if analysis_scope:
        st.markdown("### 📋 Phase 1: Planning & Analysis Scope")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Analysis Priorities:**")
            priorities = analysis_scope.get("analysis_priorities", [])
            for priority in priorities:
                st.markdown(f"• {priority.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**Data Sources:**")
            sources = analysis_scope.get("data_sources", [])
            for source in sources:
                st.markdown(f"• {source.replace('_', ' ').title()}")
        
        st.markdown(f"**Analysis Depth:** {analysis_scope.get('analysis_depth', 'N/A').title()}")
        st.markdown(f"**Time Horizon:** {analysis_scope.get('time_horizon', 'N/A').replace('_', ' ').title()}")
        st.markdown("---")
    
    # Evidence Gathering
    evidence = results.get("evidence", {})
    if evidence:
        st.markdown("### 📊 Phase 2: Evidence Gathering")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            news_count = len(evidence.get("news", []))
            st.metric("News Articles", news_count)
        
        with col2:
            price_data = evidence.get("price_data", {})
            price_count = len(price_data.get("data", [])) if price_data else 0
            st.metric("Price Data Points", price_count)
        
        with col3:
            earnings_data = evidence.get("earnings", [])
            earnings_count = len(earnings_data) if earnings_data else 0
            st.metric("Earnings Records", earnings_count)
        
        st.markdown("---")
    
    # Signal Generation
    signals = results.get("signals", {})
    if signals:
        st.markdown("### 🎯 Phase 3: Signal Generation & Analysis")
        
        signal_types = [
            ("News Impact", signals.get("news_impact", [])),
            ("Technical Analysis", signals.get("technical", [])),
            ("Earnings Analysis", signals.get("earnings", [])),
            ("Regulatory Analysis", signals.get("regulatory", [])),
            ("Governance Analysis", signals.get("governance", []))
        ]
        
        cols = st.columns(5)
        for i, (signal_type, signal_data) in enumerate(signal_types):
            with cols[i]:
                count = len(signal_data) if signal_data else 0
                st.metric(signal_type, count)
        
        st.markdown("---")
    
    # Evaluation Results
    evaluation = results.get("evaluation", {})
    if evaluation:
        st.markdown("### 🔍 Phase 4: Evaluation & Quality Assessment")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            score = evaluation.get("score", 0)
            st.metric("Overall Score", f"{score:.1%}")
        
        with col2:
            completeness = evaluation.get("completeness", 0)
            st.metric("Completeness", f"{completeness:.1%}")
        
        with col3:
            consistency = evaluation.get("consistency", 0)
            st.metric("Consistency", f"{consistency:.1%}")
        
        needs_optimization = evaluation.get("needs_optimization", False)
        st.markdown(f"**Needs Optimization:** {'Yes' if needs_optimization else 'No'}")
        
        recommendations = evaluation.get("recommendations", [])
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations[:3]:
                st.markdown(f"• {rec}")
        
        st.markdown("---")
    
    # Self-Reflection Details
    reflection = results.get("reflection", {})
    if reflection:
        st.markdown("### 🤔 Phase 5: Self-Reflection & Learning")
        
        strengths = reflection.get("strengths", [])
        weaknesses = reflection.get("weaknesses", [])
        improvements = reflection.get("improvements", [])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Identified Strengths:**")
            for strength in strengths:
                st.markdown(f"✅ {strength}")
        
        with col2:
            st.markdown("**Areas for Improvement:**")
            for improvement in improvements:
                st.markdown(f"🔄 {improvement}")
        
        if weaknesses:
            st.markdown("**Identified Weaknesses:**")
            for weakness in weaknesses:
                st.markdown(f"⚠️ {weakness}")
    
    # Memory Integration
    st.markdown("### 🧠 Persistent Memory Integration")
    st.info("""
    **Memory Features:**
    - Key insights stored for future analysis
    - Success patterns identified and retained
    - Learning points integrated for continuous improvement
    - Historical analysis data preserved
    """)

def display_news_analysis(processed_items):
    """Display news analysis results."""
    
    if not processed_items:
        st.warning("No news analysis results available.")
        return
    
    for i, item in enumerate(processed_items):
        with st.expander(f"📄 {item.get('title', 'Untitled')}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Summary:**")
                st.write(item.get('summary', 'No summary available'))
                
                st.write("**Keywords:**")
                keywords = item.get('keywords', [])
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No keywords extracted")
                
                st.write("**Topic:**")
                st.write(item.get('topic', 'Unknown'))
                
                st.write("**Numbers:**")
                numbers = item.get('numbers', [])
                if numbers:
                    st.write(", ".join(numbers))
                else:
                    st.write("No numbers extracted")
            
            with col2:
                # Sentiment
                impact = item.get('impact', {})
                sentiment = impact.get('sentiment', 'neutral')
                score = impact.get('score', 0)
                sentiment_class = sentiment
                st.markdown(f"**Sentiment:** <span class='sentiment-{sentiment_class}'>{sentiment} ({score:.2f})</span>", unsafe_allow_html=True)
                
                # Publisher
                st.write(f"**Publisher:** {item.get('publisher', 'Unknown')}")
                
                # URL
                url = item.get('link')
                if url:
                    st.write(f"**Source:** [Read More]({url})")

def display_key_findings(analysis):
    """Display key findings from the analysis."""
    
    if not analysis:
        st.warning("No analysis results available.")
        return
    
    # Display summary
    st.markdown(create_gradient_header("Analysis Summary", "📋", "Comprehensive analysis overview"), unsafe_allow_html=True)
    st.write(analysis.get('summary', 'No summary available'))
    
    # Key findings
    key_findings = analysis.get('key_findings', [])
    if key_findings:
        st.subheader(f"🔍 Key Findings ({len(key_findings)})")
        for i, finding in enumerate(key_findings, 1):
            with st.expander(f"Finding {i}: {finding.get('description', 'No description')}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {finding.get('type', 'Unknown')}")
                with col2:
                    st.write(f"**Confidence:** {finding.get('confidence', 0):.2f}")
    
    # Risk factors
    risk_factors = analysis.get('risk_factors', [])
    if risk_factors:
        st.subheader(f"⚠️ Risk Factors ({len(risk_factors)})")
        for i, risk in enumerate(risk_factors, 1):
            st.write(f"{i}. {risk}")
    
    # Opportunities
    opportunities = analysis.get('opportunities', [])
    if opportunities:
        st.subheader(f"💡 Opportunities ({len(opportunities)})")
        for i, opp in enumerate(opportunities, 1):
            st.write(f"{i}. {opp}")

def display_visualizations(processed_items):
    """Display visualizations of the analysis results."""
    
    if not processed_items:
        st.warning("No data available for visualization.")
        return
    
    # Sentiment-based visualizations section
    st.markdown(create_gradient_header("Sentiment-Based Article Visualization", "📈", "Interactive sentiment analysis and visualization"), unsafe_allow_html=True)
    
    # Prepare sentiment data
    sentiment_data = []
    for i, item in enumerate(processed_items):
        impact = item.get('impact', {})
        sentiment_data.append({
            'Article': f"Article {i+1}",
            'Title': item.get('title', 'Untitled')[:50] + '...' if len(item.get('title', '')) > 50 else item.get('title', 'Untitled'),
            'Sentiment': impact.get('sentiment', 'neutral'),
            'Score': impact.get('score', 0),
            'Topic': item.get('topic', 'Unknown'),
            'Publisher': item.get('publisher', 'Unknown')
        })
    
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Sentiment distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = sentiment_df['Sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Articles by Sentiment",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#808080'
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Sentiment Score Distribution")
        fig_hist = px.histogram(
            sentiment_df,
            x='Score',
            nbins=20,
            title="Distribution of Sentiment Scores",
            color='Sentiment',
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#808080'
            }
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Sentiment by topic analysis
    st.subheader("🏷️ Sentiment by Topic")
    
    # Create sentiment-topic cross-tabulation
    topic_sentiment = pd.crosstab(sentiment_df['Topic'], sentiment_df['Sentiment'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_stacked = px.bar(
            topic_sentiment,
            title="Sentiment Distribution by Topic",
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C', 
                'neutral': '#808080'
            }
        )
        fig_stacked.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_stacked, use_container_width=True)
    
    with col2:
        # Sentiment heatmap by topic
        fig_heatmap = px.imshow(
            topic_sentiment.T,
            labels=dict(x="Topic", y="Sentiment", color="Count"),
            title="Sentiment-Topic Heatmap",
            color_continuous_scale="RdYlGn"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Interactive sentiment scatter plot
    st.subheader("🎯 Interactive Sentiment Analysis")
    
    # Add hover information
    # Create a size column with absolute values to avoid negative size errors
    sentiment_df['Size'] = abs(sentiment_df['Score'])
    
    fig_scatter = px.scatter(
        sentiment_df,
        x='Score',
        y='Topic',
        color='Sentiment',
        size='Size',
        hover_data=['Title', 'Publisher'],
        title="Article Sentiment Scores by Topic",
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C', 
            'neutral': '#808080'
        }
    )
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Sentiment timeline (if datetime available)
    datetime_items = [item for item in processed_items if item.get('datetime')]
    if datetime_items:
        st.subheader("⏰ Sentiment Timeline")
        
        timeline_data = []
        for item in datetime_items:
            try:
                # Try to parse datetime
                dt = pd.to_datetime(item.get('datetime'))
                impact = item.get('impact', {})
                timeline_data.append({
                    'DateTime': dt,
                    'Sentiment': impact.get('sentiment', 'neutral'),
                    'Score': impact.get('score', 0),
                    'Title': item.get('title', 'Untitled')[:30] + '...'
                })
            except:
                continue
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('DateTime')
            
            fig_timeline = px.scatter(
                timeline_df,
                x='DateTime',
                y='Score',
                color='Sentiment',
                hover_data=['Title'],
                title="Sentiment Over Time",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'negative': '#DC143C', 
                    'neutral': '#808080'
                }
            )
            fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Summary statistics
    st.subheader("📊 Sentiment Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = sentiment_df['Score'].mean()
        st.metric("Average Sentiment Score", f"{avg_score:.2f}")
    
    with col2:
        positive_count = len(sentiment_df[sentiment_df['Sentiment'] == 'positive'])
        st.metric("Positive Articles", positive_count)
    
    with col3:
        negative_count = len(sentiment_df[sentiment_df['Sentiment'] == 'negative'])
        st.metric("Negative Articles", negative_count)
    
    with col4:
        neutral_count = len(sentiment_df[sentiment_df['Sentiment'] == 'neutral'])
        st.metric("Neutral Articles", neutral_count)
    
    # Content type distribution
    content_types = [item.get('topic', 'Unknown') for item in processed_items]
    type_counts = pd.Series(content_types).value_counts()
    
    st.subheader("📋 Content Type Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Distribution of Content Types"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Content Types Count"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Keyword cloud (simplified)
    all_keywords = []
    for item in processed_items:
        keywords = item.get('keywords', [])
        all_keywords.extend(keywords)
    
    if all_keywords:
        keyword_counts = pd.Series(all_keywords).value_counts().head(20)
        
        st.subheader("🔤 Top Keywords")
        fig_keywords = px.bar(
            x=keyword_counts.values,
            y=keyword_counts.index,
            orientation='h',
            title="Most Frequent Keywords"
        )
        fig_keywords.update_layout(height=500)
        st.plotly_chart(fig_keywords, use_container_width=True)

def display_detailed_results(results):
    """Display detailed analysis results."""
    
    st.markdown(create_gradient_header("Detailed Analysis Results", "🔍", "Comprehensive analysis breakdown"), unsafe_allow_html=True)
    
    # Evaluation results
    eval_results = results.get('evaluation', {})
    if eval_results:
        st.write("**Evaluation Results:**")
        st.json(eval_results)
    
    # Analysis summary
    analysis = results.get('analysis', {})
    if analysis:
        st.write("**Analysis Summary:**")
        st.json(analysis)
    
    # Signals
    signals = results.get('signals', {})
    if signals:
        st.write("**Signals Detected:**")
        for signal_type, signal_list in signals.items():
            if isinstance(signal_list, list) and signal_list:
                st.write(f"**{signal_type.replace('_', ' ').title()}:**")
                for i, signal in enumerate(signal_list[:5], 1):  # Show first 5 signals
                    with st.expander(f"Signal {i}: {signal.get('type', 'Unknown')}", expanded=False):
                        st.json(signal)
    
    # Evidence
    evidence = results.get('evidence', {})
    if evidence:
        st.write("**Evidence Sources:**")
        for source_type, source_data in evidence.items():
            if source_data:
                st.write(f"**{source_type.replace('_', ' ').title()}:**")
                if isinstance(source_data, dict):
                    # Show summary for complex data
                    if 'summary' in source_data:
                        st.json(source_data['summary'])
                    else:
                        st.write(f"Data available: {len(source_data)} items")
                elif isinstance(source_data, list):
                    st.write(f"Data available: {len(source_data)} items")
                else:
                    st.write("Data available")
    
    # Reflection
    reflection = results.get('reflection', {})
    if reflection:
        st.write("**Self-Reflection:**")
        st.json(reflection)

def safe_json_serialize(obj):
    """Safely serialize objects for JSON, handling pandas Timestamps and other non-serializable types."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, date
    
    # Handle pandas Timestamp specifically
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # Handle datetime objects
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    # Handle numpy datetime64
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    # Handle numpy arrays
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle numpy scalars
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {str(key): safe_json_serialize(value) for key, value in obj.items()}
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    # Handle sets
    elif isinstance(obj, set):
        return [safe_json_serialize(item) for item in obj]
    # Handle other objects with isoformat method
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    # Handle other objects with tolist method
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    # Handle other objects with item method
    elif hasattr(obj, 'item'):
        return obj.item()
    # Return as-is for basic types
    else:
        return obj

def display_raw_data(results):
    """Display raw data from the analysis."""
    
    st.markdown(create_gradient_header("Raw Analysis Data", "💾", "Complete analysis data in JSON format"), unsafe_allow_html=True)
    
    try:
        # Safely serialize results for display
        safe_results = safe_json_serialize(results)
        
        # Display as JSON
        st.json(safe_results)
        
        # Download button
        json_str = json.dumps(safe_results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Results as JSON",
            data=json_str,
            file_name=f"analysis_results_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error serializing data: {e}")
        st.write("**Raw data (fallback display):**")
        st.write(results)

if __name__ == "__main__":
    main()
