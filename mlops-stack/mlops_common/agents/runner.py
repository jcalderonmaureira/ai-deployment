"""
runner.py
─────────
Orquestador de ejecución del sistema multi-agente.

Instancia los 6 agentes, los conecta al EventBus, y ejecuta ciclos
de observe → decide → act en secuencia, demostrando el flujo completo
de 10 pasos del sistema (clase-multiagente-mlops.html, Parte 05).

Uso:
  # Demostración con auto-approve (sin interacción humana)
  python -m mlops_common.agents.runner --auto-approve

  # Con aprobación manual (se detiene para esperar input del humano)
  python -m mlops_common.agents.runner

  # Dentro de Docker:
  docker compose run --rm model-trainer python -m mlops_common.agents.runner
"""

import argparse
import logging
import time
import sys

from .event_bus import EventBus
from .orchestrator_agent import OrchestratorAgent
from .data_scientist_agent import DataScientistAgent
from .ml_engineer_agent import MLEngineerAgent
from .qa_agent import QAAgent
from .monitoring_agent import MonitoringAgent
from .human_proxy_agent import HumanProxyAgent


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)-20s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )


def create_system(
    auto_approve: bool = False,
    mlflow_uri: str = "http://mlflow:5001",
    api_url: str = "http://inference-api:8000",
    drift_threshold: int = 2,
):
    """Crea el sistema multi-agente completo, conectado vía EventBus."""

    bus = EventBus()

    orchestrator = OrchestratorAgent(bus, auto_approve=auto_approve)
    ds = DataScientistAgent(bus, mlflow_tracking_uri=mlflow_uri)
    mle = MLEngineerAgent(bus, inference_api_url=api_url)
    qa = QAAgent(bus)
    mon = MonitoringAgent(
        bus,
        drift_threshold=drift_threshold,
        inference_api_url=api_url,
        mlflow_tracking_uri=mlflow_uri,
    )
    human = HumanProxyAgent(bus, auto_mode=auto_approve)

    return {
        "bus": bus,
        "orchestrator": orchestrator,
        "data_scientist": ds,
        "ml_engineer": mle,
        "qa_engineer": qa,
        "monitoring": mon,
        "human_proxy": human,
    }


def run_demo(system: dict, cycles: int = 5, interval: float = 2.0):
    """Ejecuta N ciclos del sistema multi-agente.

    Cada ciclo ejecuta el ciclo observe→decide→act de cada agente en
    el orden natural del pipeline: MON → Orchestrator → DS → QA → Human → MLE.
    """
    mon = system["monitoring"]
    orch = system["orchestrator"]
    ds = system["data_scientist"]
    qa = system["qa_engineer"]
    human = system["human_proxy"]
    mle = system["ml_engineer"]

    agent_order = [mon, orch, ds, qa, human, mle]
    logger = logging.getLogger("runner")

    for cycle in range(1, cycles + 1):
        logger.info("═" * 60)
        logger.info("CYCLE %d / %d  (orchestrator state: %s)", cycle, cycles, orch.state)
        logger.info("═" * 60)

        for agent in agent_order:
            result = agent.run_cycle()
            if result.detail != "No action needed":
                logger.info("[%s] %s", agent.name, result.detail)

        # Si el humano tiene una aprobación pendiente y no es auto-mode
        if human.has_pending_approval and not human.auto_mode:
            logger.info("─" * 40)
            logger.info("HUMAN DECISION REQUIRED")
            logger.info("Model details: %s", human.pending_details)
            try:
                choice = input("  Approve? [y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                logger.info("Non-interactive — skipping human approval")
                choice = "n"

            if choice == "y":
                human.approve()
                orch.approve()
            else:
                reason = "Rejected by human operator"
                human.reject(reason)
                orch.reject(reason)

        if interval > 0:
            time.sleep(interval)

    # Resumen final
    bus = system["bus"]
    logger.info("═" * 60)
    logger.info("DEMO COMPLETE — %d events in history:", len(bus.history))
    for msg in bus.history:
        logger.info("  [%s] %s → %s", msg.sender, msg.topic, msg.payload)


def main():
    parser = argparse.ArgumentParser(description="MLOps Multi-Agent System Runner")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Auto-approve models (no human intervention)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of cycles to run (default: 5)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between cycles (default: 2.0)")
    parser.add_argument("--mlflow-uri", default="http://mlflow:5001")
    parser.add_argument("--api-url", default="http://inference-api:8000")
    parser.add_argument("--drift-threshold", type=int, default=2)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    system = create_system(
        auto_approve=args.auto_approve,
        mlflow_uri=args.mlflow_uri,
        api_url=args.api_url,
        drift_threshold=args.drift_threshold,
    )

    run_demo(system, cycles=args.cycles, interval=args.interval)


if __name__ == "__main__":
    main()
