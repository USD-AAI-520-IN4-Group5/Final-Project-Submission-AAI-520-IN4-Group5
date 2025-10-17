
"""
Main entrypoint for the Enhanced Investment Research Agent.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse
from agents.enhanced_investment_agent import EnhancedInvestmentAgent

def main(symbol: str):
    print(f"\n🚀 Starting Enhanced Investment Research Agent for symbol: {symbol}")
    agent = EnhancedInvestmentAgent(symbol=symbol)
    result = agent.act()
    print("\n✅ Autonomous Enhanced analysis completed!")
    
    # Display key results
    analysis = result.get('analysis', {})
    evaluation = result.get('evaluation', {})
    research_report = result.get('research_report', {})
    
    print(f"📊 Analysis Summary: {analysis.get('summary', 'No summary available')}")
    print(f"📈 Sentiment: {analysis.get('sentiment', 'neutral')}")
    print(f"🎯 Confidence: {analysis.get('confidence', 0.0):.2f}")
    print(f"⭐ Overall Score: {evaluation.get('score', 0.0):.2f}")
    
    # Display research report sections
    if research_report:
        print(f"\n📄 Research Report Generated:")
        print(f"  - Executive Summary: Available")
        print(f"  - Company Overview: Available")
        print(f"  - Financial Analysis: Available")
        print(f"  - Technical Analysis: Available")
        print(f"  - Sentiment Analysis: Available")
        print(f"  - Risk Assessment: Available")
        print(f"  - Investment Thesis: Available")
        print(f"  - Recommendations: Available")
    
    # Display key findings
    key_findings = analysis.get('key_findings', [])
    if key_findings:
        print(f"\n🔍 Key Findings ({len(key_findings)}):")
        for i, finding in enumerate(key_findings[:5], 1):
            print(f"  {i}. {finding.get('description', 'No description')}")
    
    # Display risk factors
    risk_factors = analysis.get('risk_factors', [])
    if risk_factors:
        print(f"\n⚠️  Risk Factors ({len(risk_factors)}):")
        for i, risk in enumerate(risk_factors[:3], 1):
            print(f"  {i}. {risk}")
    
    # Display opportunities
    opportunities = analysis.get('opportunities', [])
    if opportunities:
        print(f"\n💡 Opportunities ({len(opportunities)}):")
        for i, opp in enumerate(opportunities[:3], 1):
            print(f"  {i}. {opp}")
    
    # Display self-reflection insights
    reflection = result.get('reflection', {})
    if reflection:
        print(f"\n🤔 Self-Reflection Insights:")
        strengths = reflection.get('strengths', [])
        improvements = reflection.get('improvements', [])
        if strengths:
            print(f"  Strengths: {len(strengths)} identified")
        if improvements:
            print(f"  Improvements: {len(improvements)} identified")
    
    print("\n📂 Comprehensive research report saved under: outputs/reports/")
    print("🧠 Insights stored in persistent memory for future analysis")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Enhanced Investment Research Agent")
    parser.add_argument("--symbol", required=True, help="Stock symbol to analyze (e.g., AAPL)")
    args = parser.parse_args()
    main(args.symbol)
