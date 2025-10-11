# Enhanced Investment Research Agent

A comprehensive Enhanced investment research agent capable of autonomously conducting financial analysis for any given stock ticker. The agent dynamically retrieves and processes diverse sources of financial and market data, synthesizing this information into coherent investment research reports.

## Problem Statement

The objective of this project is to design and implement a Enhanced investment research agent capable of autonomously conducting financial analysis for a given stock ticker. The agent must dynamically retrieve and process diverse sources of financial and market data—including historical prices, company fundamentals, earnings reports, and news articles—and synthesize this information into a coherent investment research report.

The workflow involves the agent planning and executing a sequence of analytical tasks, such as evidence gathering, data preprocessing, report generation, optimization, and evaluation. Additionally, the system must demonstrate self-reflection by assessing the quality of its outputs and retaining relevant insights in persistent memory for future runs.

## Key Features

- 🤖 **Autonomous Enhanced Agent**: Self-directed financial analysis with predefined rules and workflows
- 📊 **Dynamic Data Retrieval**: Multi-source data gathering from Yahoo Finance, News API, and other financial sources
- 🧠 **Intelligent Processing**: Advanced LLM-based sentiment analysis, topic classification, and content filtering
- 📈 **Comprehensive Analysis**: Technical analysis, fundamental analysis, earnings analysis, and news impact assessment
- 🔄 **Self-Reflection**: Quality assessment and optimization with persistent memory
- 📋 **Coherent Reporting**: Structured investment research reports with actionable insights
- 🎯 **Risk Assessment**: Automated identification of risk factors and investment opportunities

## System Architecture

```
src/
├── agents/                    # Core agent implementations
│   ├── base_agent.py         # Base agent class with common functionality
│   ├── investment_agent.py   # Main investment research agent
│   ├── enhanced_investment_agent.py  # Enhanced agent with advanced capabilities
│   ├── earnings_agent.py     # Specialized earnings analysis agent
│   ├── news_agent.py        # News processing and analysis agent
│   ├── evaluator_agent.py   # Quality evaluation and optimization agent
│   └── specialist_agents.py # Specialized analysis agents (technical, regulatory, etc.)
├── tools/                    # Data retrieval and processing tools
│   ├── news_api.py          # News API integration
│   ├── yfinance_client.py   # Yahoo Finance data client
│   └── memory_store.py      # Persistent memory management
├── workflows/               # Analysis workflow implementations
│   ├── prompt_chaining.py   # Sequential task execution
│   ├── routing.py          # Content routing to specialist agents
│   └── evaluator_optimizer.py  # Quality optimization workflows
└── utils/                  # Utility functions and helpers
    ├── logger.py           # Logging configuration
    ├── text_processing.py  # Advanced text analysis utilities
    └── config_loader.py    # Configuration management
```

## Core Workflow

The investment research agent follows a structured workflow:

### 1. Planning Phase
- **Task Decomposition**: Break down analysis into manageable subtasks
- **Resource Allocation**: Determine data sources and analysis priorities
- **Timeline Estimation**: Plan execution sequence and dependencies

### 2. Execution Phase
- **Evidence Gathering**: Retrieve historical prices, fundamentals, earnings, news
- **Data Preprocessing**: Clean, validate, and structure incoming data
- **Analysis Processing**: Apply specialized analysis rules and algorithms
- **Signal Generation**: Extract actionable insights and patterns

### 3. Synthesis Phase
- **Report Generation**: Compile findings into coherent investment research report
- **Risk Assessment**: Identify potential risks and opportunities
- **Recommendation Formulation**: Generate investment recommendations

### 4. Evaluation Phase
- **Quality Assessment**: Evaluate report completeness and accuracy
- **Self-Reflection**: Analyze performance and identify improvement areas
- **Optimization**: Refine analysis based on evaluation results

### 5. Memory Phase
- **Knowledge Retention**: Store insights in persistent memory
- **Learning Integration**: Update rules and patterns for future runs

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd financial-agentic-ai-final
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n usd-aai python=3.9
   conda activate usd-aai
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### API Keys Setup

1. **News API**: Get a free API key from [News API](https://newsapi.org/)
2. **OpenAI API** (Optional): For advanced LLM-based analysis
3. **Hugging Face Token** (Optional): For transformer model access

### Setting Up API Keys

**Option 1: Through the Web Interface (Recommended)**
1. Run the Streamlit app: `streamlit run app.py`
2. Enter your API keys in the "API Configuration" section
3. The keys will be used for that session only

**Option 2: Through Configuration File**
Update `config.yaml` with your API keys:
```yaml
api:
  news_api_key: your_news_api_key
  openai_api_key: your_openai_key  # Optional
  huggingface_token: your_hf_token  # Optional
```

## Usage

### Command Line Interface

Run autonomous analysis for a specific stock symbol:

```bash
python main.py --symbol AAPL
```

### Web Interface (Streamlit)

Launch the interactive research dashboard:

```bash
streamlit run app.py
```

The web interface provides:
- 📈 Real-time analysis dashboard
- 📊 Interactive visualizations and charts
- 📰 News article analysis and sentiment tracking
- 🔍 Detailed research findings exploration
- 📥 Export capabilities for reports

## Analysis Capabilities

### Data Sources
- **Historical Price Data**: OHLCV data, technical indicators
- **Company Fundamentals**: Financial statements, ratios, metrics
- **Earnings Reports**: Quarterly results, guidance, analyst estimates
- **News Articles**: Real-time financial news with sentiment analysis
- **Market Data**: Sector performance, economic indicators

### Analysis Types
- **Technical Analysis**: Moving averages, RSI, Bollinger Bands, volatility
- **Fundamental Analysis**: Financial ratios, growth metrics, valuation
- **Sentiment Analysis**: News sentiment, market sentiment, social sentiment
- **Risk Assessment**: Systematic risk, company-specific risk factors
- **Opportunity Identification**: Growth catalysts, market inefficiencies

### Report Components
- **Executive Summary**: Key findings and recommendations
- **Company Overview**: Business model and competitive position
- **Financial Analysis**: Performance metrics and trends
- **Risk Factors**: Identified risks and mitigation strategies
- **Investment Thesis**: Buy/Hold/Sell recommendation with rationale

## Output Files

Analysis results are systematically saved to:
- `outputs/reports/{SYMBOL}_enhanced_analysis.json` - Complete research report
- `enhanced_runs.jsonl` - Persistent memory and learning data
- `yahoo_cache/` - Cached market data for performance optimization

## Self-Reflection and Learning

The agent demonstrates self-reflection through:

### Quality Assessment
- **Completeness Scoring**: Evaluate report comprehensiveness
- **Accuracy Validation**: Cross-reference findings with market data
- **Consistency Checks**: Ensure logical coherence across analysis

### Performance Analysis
- **Execution Time**: Monitor analysis efficiency
- **Resource Utilization**: Track API usage and computational costs
- **Error Analysis**: Identify and learn from failure patterns

### Memory Integration
- **Insight Retention**: Store valuable findings for future reference
- **Pattern Recognition**: Learn from successful analysis patterns
- **Rule Refinement**: Update analysis rules based on performance

## Examples

### Analyze Apple Stock (AAPL)
```bash
python main.py --symbol AAPL
```

### Web Interface Analysis
1. Open `http://localhost:8501`
2. Enter stock symbol (e.g., "AAPL", "MSFT", "GOOGL")
3. Configure analysis parameters
4. Click "Run Analysis"
5. Explore comprehensive research report

### Sample Analysis Output
```json
{
  "symbol": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "analysis": {
    "summary": "Strong fundamentals with positive sentiment",
    "sentiment": "positive",
    "confidence": 0.85,
    "overall_score": 0.72,
    "key_findings": [...],
    "risk_factors": [...],
    "opportunities": [...]
  },
  "evaluation": {
    "score": 0.78,
    "completeness": 0.92,
    "needs_optimization": false
  },
  "reflection": {
    "strengths": [...],
    "improvements": [...]
  }
}
```

## Advanced Features

### Multi-Agent Architecture
- **Specialist Agents**: Dedicated agents for earnings, technical, regulatory analysis
- **Coordinated Workflow**: Seamless handoff between specialized agents
- **Parallel Processing**: Concurrent analysis for improved efficiency

### Intelligent Filtering
- **LLM-Based Relevance**: Advanced content filtering using transformer models
- **Context Awareness**: Understanding of financial vs. non-financial content
- **Quality Control**: Automated filtering of low-quality or irrelevant data

### Persistent Learning
- **Memory System**: Long-term storage of analysis insights
- **Pattern Recognition**: Learning from historical analysis patterns
- **Adaptive Rules**: Dynamic rule updates based on performance

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Adjust request frequency and implement caching
2. **Memory Issues**: Reduce analysis scope or increase system resources
3. **Import Errors**: Ensure all dependencies are properly installed
4. **Data Quality**: Verify API keys and data source availability

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement Enhanced enhancements
4. Add comprehensive tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Future Enhancements

- [ ] Additional data sources (SEC filings, analyst reports)
- [ ] Advanced ML models for pattern recognition
- [ ] Real-time market monitoring and alerts
- [ ] Portfolio-level analysis capabilities
- [ ] Integration with trading platforms
- [ ] Enhanced risk modeling and stress testing

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the comprehensive documentation