"""
agents
──────
Framework agéntico rule-based para el stack MLOps.

5 agentes + Human-in-the-Loop, comunicados vía EventBus (pub/sub),
mapeados 1:1 a los roles de Eken et al. (2025) §3.3 y a los servicios
Docker Compose del stack.
"""

from .base import AgentBase, AgentMessage, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus
from .orchestrator_agent import OrchestratorAgent
from .data_scientist_agent import DataScientistAgent
from .ml_engineer_agent import MLEngineerAgent
from .qa_agent import QAAgent
from .monitoring_agent import MonitoringAgent
from .human_proxy_agent import HumanProxyAgent

AGENT_REGISTRY = {
    "orchestrator":   OrchestratorAgent,
    "data_scientist": DataScientistAgent,
    "ml_engineer":    MLEngineerAgent,
    "qa_engineer":    QAAgent,
    "monitoring":     MonitoringAgent,
    "human_proxy":    HumanProxyAgent,
}

__all__ = [
    "AgentBase", "AgentMessage", "Observation", "Decision", "ActionResult",
    "Topics", "EventBus",
    "OrchestratorAgent", "DataScientistAgent", "MLEngineerAgent",
    "QAAgent", "MonitoringAgent", "HumanProxyAgent",
    "AGENT_REGISTRY",
]
