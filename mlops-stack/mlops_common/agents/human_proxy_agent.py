"""
human_proxy_agent.py
────────────────────
Agente Human-in-the-Loop — Eken et al. §5.2.5 (Automation Dilemma).

No es un servicio Docker — es un punto deliberado de aprobación manual.
Actúa como proxy entre el sistema de agentes y el humano que decide
si promover o rechazar un modelo candidato.

Suscrito a: approval.requested
Publica:    (delega al OrchestratorAgent via approve()/reject())
"""

from typing import Any, Dict, Optional

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class HumanProxyAgent(AgentBase):
    """Proxy del humano en el sistema de agentes.

    Acumula las solicitudes de aprobación y las expone para que el humano
    pueda decidir. El mecanismo de decisión es externo:
      - CLI interactivo (para desarrollo)
      - HTTP endpoint (para integración con UI)
      - Webhook (para Slack/Teams/email)
      - Auto-approve (para tests automatizados)
    """

    def __init__(self, bus: EventBus, auto_mode: bool = False):
        super().__init__(name="human_proxy", role="HUMAN", bus=bus)
        self.auto_mode = auto_mode
        self._pending_approval: Optional[Dict[str, Any]] = None
        self.subscribe(Topics.APPROVAL_REQUESTED)

    def on_message(self, message):
        super().on_message(message)
        if message.topic == Topics.APPROVAL_REQUESTED:
            self._pending_approval = message.payload
            self.logger.info(
                "Approval request received — metrics: %s",
                message.payload.get("metrics"),
            )

    @property
    def has_pending_approval(self) -> bool:
        return self._pending_approval is not None

    @property
    def pending_details(self) -> Optional[Dict[str, Any]]:
        return self._pending_approval

    def observe(self) -> Observation:
        return Observation(data={
            "pending": self._pending_approval is not None,
            "details": self._pending_approval,
        })

    def decide(self, observation: Observation) -> Decision:
        if not observation.data["pending"]:
            return Decision(action="noop", reason="No pending approvals")

        if self.auto_mode:
            return Decision(
                action="auto_approve",
                reason="auto_mode=True — approving automatically",
                params=observation.data["details"] or {},
            )

        return Decision(
            action="wait",
            reason="Waiting for human decision (approve/reject)",
            params=observation.data["details"] or {},
        )

    def act(self, decision: Decision) -> ActionResult:
        if decision.action == "auto_approve":
            self._pending_approval = None
            return ActionResult(
                success=True,
                detail="Auto-approved (test mode)",
            )

        if decision.action == "wait":
            return ActionResult(
                success=True,
                detail="Awaiting human input — call approve() or reject()",
            )

        return ActionResult(success=True, detail="No action")

    def approve(self):
        """El humano aprueba el modelo candidato."""
        if not self._pending_approval:
            self.logger.warning("No pending approval to approve")
            return
        self.logger.info("HUMAN APPROVED model")
        self._pending_approval = None

    def reject(self, reason: str = ""):
        """El humano rechaza el modelo candidato."""
        if not self._pending_approval:
            self.logger.warning("No pending approval to reject")
            return
        self.logger.info("HUMAN REJECTED model: %s", reason)
        self._pending_approval = None
