"""
orchestrator_agent.py
─────────────────────
Agente MLOE (Orquestador) — Eken et al. §3.3.

Servicio Docker: nginx + gestión de GitHub Actions workflows.
Responsabilidad: coordinar a todos los demás agentes, gestionar el estado
global del pipeline, y escalar al humano cuando se necesita aprobación.

Suscrito a: drift.detected, qa.passed, qa.failed, model.reloaded, system.error
Publica:    retrain.triggered, approval.requested, model.promoted
"""

from typing import Any, Dict

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class OrchestratorAgent(AgentBase):
    """Agente central que coordina el flujo de los 10 pasos del sistema.

    Mantiene un estado de pipeline (idle → retraining → qa → approval → promoting)
    y reacciona a los mensajes de los demás agentes.
    """

    STATES = ("idle", "retraining", "qa_pending", "approval_pending", "promoting")

    def __init__(self, bus: EventBus, auto_approve: bool = False):
        super().__init__(name="orchestrator", role="MLOE", bus=bus)
        self.state = "idle"
        self.auto_approve = auto_approve
        self._pending_events: list = []

        self.subscribe(Topics.DRIFT_DETECTED)
        self.subscribe(Topics.QA_PASSED)
        self.subscribe(Topics.QA_FAILED)
        self.subscribe(Topics.MODEL_RELOADED)
        self.subscribe(Topics.SYSTEM_ERROR)

    def on_message(self, message):
        super().on_message(message)
        self._pending_events.append(message)

    def observe(self) -> Observation:
        events = list(self._pending_events)
        self._pending_events.clear()
        return Observation(data={
            "state": self.state,
            "events": [
                {"topic": m.topic, "payload": m.payload, "sender": m.sender}
                for m in events
            ],
        })

    def decide(self, observation: Observation) -> Decision:
        """Máquina de estados del pipeline."""
        events = observation.data["events"]

        for event in events:
            topic = event["topic"]

            if topic == Topics.DRIFT_DETECTED and self.state == "idle":
                return Decision(
                    action="trigger_retrain",
                    reason=f"Drift detected: {event['payload'].get('reason', '')}",
                    params=event["payload"],
                )

            if topic == Topics.QA_PASSED and self.state == "qa_pending":
                if self.auto_approve:
                    return Decision(
                        action="promote",
                        reason="QA passed + auto_approve=True",
                        params=event["payload"],
                    )
                return Decision(
                    action="request_approval",
                    reason="QA passed — requesting human approval",
                    params=event["payload"],
                )

            if topic == Topics.QA_FAILED and self.state == "qa_pending":
                self.state = "idle"
                return Decision(
                    action="noop",
                    reason=f"QA failed: {event['payload'].get('failures', '')} — pipeline stopped",
                )

            if topic == Topics.MODEL_RELOADED and self.state == "promoting":
                self.state = "idle"
                self.logger.info("Pipeline complete — model reloaded successfully")
                return Decision(action="noop", reason="Pipeline cycle complete")

            if topic == Topics.SYSTEM_ERROR:
                self.logger.error("System error from %s: %s",
                                  event["sender"], event["payload"])

        return Decision(action="noop", reason=f"State={self.state}, no actionable events")

    def act(self, decision: Decision) -> ActionResult:
        if decision.action == "trigger_retrain":
            self.state = "retraining"
            self.publish(Topics.RETRAIN_TRIGGERED, {
                "reason": decision.reason,
                **decision.params,
            })
            self.state = "qa_pending"
            return ActionResult(
                success=True,
                detail="Retrain triggered → waiting for DS + QA",
            )

        if decision.action == "request_approval":
            self.state = "approval_pending"
            self.publish(Topics.APPROVAL_REQUESTED, decision.params)
            return ActionResult(
                success=True,
                detail="Approval requested — waiting for human",
            )

        if decision.action == "promote":
            self.state = "promoting"
            self.publish(Topics.MODEL_PROMOTED, decision.params)
            return ActionResult(
                success=True,
                detail="Model promoted → MLE will reload",
            )

        return ActionResult(success=True, detail=f"Unhandled action: {decision.action}")

    def approve(self):
        """Llamado externamente cuando el humano aprueba el modelo."""
        if self.state != "approval_pending":
            self.logger.warning("approve() called but state=%s", self.state)
            return
        self.state = "promoting"
        self.publish(Topics.MODEL_PROMOTED, {
            "approved_by": "human",
        })

    def reject(self, reason: str = ""):
        """Llamado externamente cuando el humano rechaza el modelo."""
        if self.state != "approval_pending":
            self.logger.warning("reject() called but state=%s", self.state)
            return
        self.state = "idle"
        self.publish(Topics.MODEL_REJECTED, {"reason": reason})
        self.logger.info("Model rejected by human: %s", reason)
