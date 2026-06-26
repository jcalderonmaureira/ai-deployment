"""
ml_engineer_agent.py
────────────────────
Agente MLE (ML Engineer) — Eken et al. §3.3.

Servicio Docker: inference-api (FastAPI :8000).
Responsabilidad: servir predicciones y ejecutar hot-swap del modelo
sin downtime cuando se promueve una nueva versión.

Suscrito a: model.promoted
Publica:    model.reloaded
"""

from typing import Any, Dict

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class MLEngineerAgent(AgentBase):
    """Agente que gestiona el modelo en producción.

    Recibe notificación de promoción y ejecuta POST /reload en inference-api
    para hot-swap sin downtime.
    """

    def __init__(
        self,
        bus: EventBus,
        inference_api_url: str = "http://inference-api:8000",
    ):
        super().__init__(name="ml_engineer", role="MLE", bus=bus)
        self.inference_api_url = inference_api_url
        self._reload_pending = False
        self._promotion_info: Dict[str, Any] = {}
        self.subscribe(Topics.MODEL_PROMOTED)

    def on_message(self, message):
        super().on_message(message)
        if message.topic == Topics.MODEL_PROMOTED:
            self._reload_pending = True
            self._promotion_info = message.payload

    def observe(self) -> Observation:
        """Verifica el estado de health del inference-api."""
        api_healthy = False
        current_version = None

        try:
            import requests
            resp = requests.get(f"{self.inference_api_url}/health", timeout=5)
            api_healthy = resp.status_code == 200

            info = requests.get(f"{self.inference_api_url}/info", timeout=5)
            if info.status_code == 200:
                current_version = info.json().get("model_version")
        except Exception:
            pass

        return Observation(data={
            "api_healthy": api_healthy,
            "current_version": current_version,
            "reload_pending": self._reload_pending,
            "promotion_info": self._promotion_info,
        })

    def decide(self, observation: Observation) -> Decision:
        if not observation.data["reload_pending"]:
            return Decision(action="noop", reason="No reload pending")

        if not observation.data["api_healthy"]:
            return Decision(
                action="noop",
                reason="API not healthy — cannot reload",
            )

        return Decision(
            action="reload",
            reason="New model promoted — hot-swap required",
            params=observation.data["promotion_info"],
        )

    def act(self, decision: Decision) -> ActionResult:
        """Ejecuta POST /reload en inference-api (hot-swap sin downtime)."""
        self._reload_pending = False
        self._promotion_info = {}

        try:
            import requests
            resp = requests.post(
                f"{self.inference_api_url}/reload", timeout=30
            )

            if resp.status_code == 200:
                body = resp.json()
                self.logger.info("Model reloaded: %s", body)
                self.publish(Topics.MODEL_RELOADED, {
                    "status": body.get("status"),
                    "version": body.get("version"),
                    "stage": body.get("stage"),
                })
                return ActionResult(
                    success=True,
                    detail=f"Model reloaded: {body}",
                    artifacts=body,
                )
            else:
                detail = f"Reload failed: HTTP {resp.status_code}"
                self.publish(Topics.SYSTEM_ERROR, {
                    "agent": self.name,
                    "error": detail,
                })
                return ActionResult(success=False, detail=detail)

        except Exception as e:
            self.logger.exception("Reload request failed")
            return ActionResult(success=False, detail=f"Reload error: {e}")
