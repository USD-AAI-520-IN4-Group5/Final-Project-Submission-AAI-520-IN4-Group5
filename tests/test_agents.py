import pytest
from src.agents.investment_agent import InvestmentResearchAgent

def test_investment_agent_runs():
    ag = InvestmentResearchAgent(symbol='TEST', max_iterations=1)
    out = ag.act()
    assert 'final' in out
    assert isinstance(out['final'], list)
