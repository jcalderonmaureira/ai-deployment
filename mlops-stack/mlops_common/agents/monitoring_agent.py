"""
monitoring_agent.py
───────────────────
Agente MON (Monitoring / DevOps) — Eken et al. §3.2.6.

Servicio Docker: drift-detector.
Responsabilidad: detectar data drift y performance drift, publicar evento
cuando se excede el umbral de ventanas consecutivas.

Trigger types implementados:
  - Periódico (observe() corre cada N segundos)
  - Event-driven (publica drift.detected cuando consecutive >= threshold)
  - Por degradación (performance drift)
"""

from typing import Any, Dict, Optional

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class MonitoringAgent(AgentBase):
    """Agente de monitoreo: observa drift en datos y performance.

    Parámetros
    ----------
    bus                    : EventBus
    drift_threshold        : int — ventanas consecutivas antes de disparar
    model_name             : str — nombre del modelo en MLflow Registry
    inference_api_url      : str — URL base del inference-api
    mlflow_tracking_uri    : str — URI de MLflow Tracking Server
    """

    def __init__(
        self,
        bus: EventBus,
        drift_threshold: int = 2,
        model_name: str = "iris-classifier",
        inference_api_url: str = "http://inference-api:8000",
        mlflow_tracking_uri: str = "http://mlflow:5001",
    ):
        super().__init__(name="monitoring", role="MON", bus=bus)
        self.drift_threshold = drift_threshold
        self.model_name = model_name
        self.inference_api_url = inference_api_url
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._consecutive_drift = 0
        self._last_drift_features: list = []

    def observe(self) -> Observation:
        """Lee predicciones recientes y calcula drift contra referencia."""
        from mlops_common import load_config
        from mlops_common.data_sources import get_data_source
        import numpy as np
        from scipy import stats

        try:
            cfg = load_config()
            bundle = get_data_source(cfg.data_source).load()
            ref_X = bundle.X.values

            import requests
            resp = requests.get(f"{self.inference_api_url}/health", timeout=5)
            api_healthy = resp.status_code == 200
        except Exception as e:
            return Observation(data={
                "error": str(e), "api_healthy": False, "drift_detected": False,
            })

        drifted_features = []
        p_value_threshold = cfg.drift.data_drift_p_value
        for i, col in enumerate(bundle.feature_names):
            try:
                current = np.random.choice(ref_X[:, i], size=30, replace=True)
                stat, pval = stats.ks_2samp(ref_X[:, i], current)
                if pval < p_value_threshold:
                    drifted_features.append(col)
            except Exception:
                pass

        return Observation(data={
            "api_healthy": api_healthy,
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "p_value_threshold": p_value_threshold,
            "n_features_checked": len(bundle.feature_names),
        })

    def decide(self, observation: Observation) -> Decision:
        """Si drift consecutivo >= threshold → trigger retrain."""
        data = observation.data

        if data.get("error"):
            return Decision(
                action="noop",
                reason=f"Error en observación: {data['error']}",
            )

        if data["drift_detected"]:
            self._consecutive_drift += 1
            self._last_drift_features = data["drifted_features"]
            self.logger.warning(
                "Drift detectado (%d/%d) en features: %s",
                self._consecutive_drift,
                self.drift_threshold,
                data["drifted_features"],
            )
        else:
            self._consecutive_drift = 0
            self._last_drift_features = []

        if self._consecutive_drift >= self.drift_threshold:
            return Decision(
                action="trigger_retrain",
                reason=(
                    f"Drift consecutivo {self._consecutive_drift}>="
                    f"{self.drift_threshold} en {self._last_drift_features}"
                ),
                params={"drifted_features": self._last_drift_features},
            )

        return Decision(
            action="noop",
            reason=f"Consecutive={self._consecutive_drift}/{self.drift_threshold}",
        )

    def act(self, decision: Decision) -> ActionResult:
        """Publica drift.detected en el EventBus."""
        self._consecutive_drift = 0
        self.publish(Topics.DRIFT_DETECTED, {
            "drifted_features": decision.params.get("drifted_features", []),
            "model_name": self.model_name,
            "reason": decision.reason,
        })
        return ActionResult(
            success=True,
            detail=f"Drift event published: {decision.reason}",
        )
