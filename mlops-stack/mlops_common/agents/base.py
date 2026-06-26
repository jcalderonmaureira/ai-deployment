"""
base.py
───────
Framework agéntico rule-based: observe → decide → act.

Cada agente del stack MLOps implementa este ciclo:
  1. observe()  → lee el estado del sistema relevante a su rol
  2. decide()   → evalúa reglas/condiciones y decide la acción
  3. act()      → ejecuta la decisión y retorna el resultado

Los agentes se comunican entre sí vía EventBus (pub/sub en memoria),
siguiendo la matriz de interacción de Eken et al. §3.3.

Referencia: clase-multiagente-mlops.html, Parte 03 "Los Cinco Agentes".
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════
# Estructuras de datos del ciclo observe → decide → act
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentMessage:
    """Mensaje inter-agente transportado por el EventBus."""
    topic: str
    payload: Dict[str, Any]
    sender: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Observation:
    """Estado del sistema observado por un agente en un instante."""
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Decision:
    """Decisión tomada por un agente tras observar.

    action: identificador de la acción ("retrain", "promote", "reload", "noop")
    reason: justificación legible (para logs y auditoría)
    params: parámetros adicionales para la acción
    """
    action: str
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionResult:
    """Resultado de ejecutar una decisión."""
    success: bool
    detail: str
    artifacts: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# Tópicos del EventBus (canales de comunicación entre agentes)
# ═══════════════════════════════════════════════════════════════

class Topics:
    """Constantes de tópicos — mapa directo a la matriz de interacción."""
    DRIFT_DETECTED = "drift.detected"
    RETRAIN_TRIGGERED = "retrain.triggered"
    MODEL_TRAINED = "model.trained"
    QA_PASSED = "qa.passed"
    QA_FAILED = "qa.failed"
    APPROVAL_REQUESTED = "approval.requested"
    MODEL_PROMOTED = "model.promoted"
    MODEL_REJECTED = "model.rejected"
    MODEL_RELOADED = "model.reloaded"
    SYSTEM_ERROR = "system.error"


# ═══════════════════════════════════════════════════════════════
# Agente base — ABC
# ═══════════════════════════════════════════════════════════════

class AgentBase(ABC):
    """Agente MLOps con ciclo observe → decide → act.

    Parámetros
    ----------
    name : str  — identificador único ("monitoring", "data_scientist", etc.)
    role : str  — rol de Eken et al. ("MLOE", "DS", "MLE", "QAE", "MON")
    bus  : EventBus  — canal de comunicación con los demás agentes
    """

    def __init__(self, name: str, role: str, bus: "EventBus"):
        self.name = name
        self.role = role
        self.bus = bus
        self.logger = logging.getLogger(f"agent.{name}")
        self._inbox: List[AgentMessage] = []

    # ── Pub/sub ──────────────────────────────────────────────────

    def subscribe(self, topic: str):
        self.bus.subscribe(topic, self._receive)

    def publish(self, topic: str, payload: Dict[str, Any]):
        msg = AgentMessage(topic=topic, payload=payload, sender=self.name)
        self.bus.publish(msg)

    def _receive(self, message: AgentMessage):
        self._inbox.append(message)
        self.logger.info(
            "Inbox ← [%s] from %s", message.topic, message.sender
        )
        self.on_message(message)

    def drain_inbox(self) -> List[AgentMessage]:
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs

    # ── Ciclo principal ──────────────────────────────────────────

    @abstractmethod
    def observe(self) -> Observation:
        """Lee el estado del sistema relevante a este agente."""
        ...

    @abstractmethod
    def decide(self, observation: Observation) -> Decision:
        """Evalúa reglas y decide la siguiente acción."""
        ...

    @abstractmethod
    def act(self, decision: Decision) -> ActionResult:
        """Ejecuta la decisión (llamadas HTTP, MLflow, tests, etc.)."""
        ...

    def on_message(self, message: AgentMessage):
        """Hook para reaccionar a mensajes entrantes (override opcional)."""
        pass

    def run_cycle(self) -> ActionResult:
        """Ejecuta un ciclo completo observe → decide → act."""
        self.logger.info("── Cycle start ──")
        obs = self.observe()
        dec = self.decide(obs)
        self.logger.info("Decision: %s — %s", dec.action, dec.reason)
        if dec.action == "noop":
            return ActionResult(success=True, detail="No action needed")
        result = self.act(dec)
        self.logger.info(
            "Result: %s — %s",
            "OK" if result.success else "FAIL",
            result.detail,
        )
        return result
