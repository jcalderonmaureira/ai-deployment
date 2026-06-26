"""
qa_agent.py
───────────
Agente QAE (Quality Assurance Engineer) — Eken et al. §3.2.8.

Servicio Docker: test-runner.
Responsabilidad: ejecutar quality gates sobre el modelo entrenado.
Actúa como puerta de control — si los tests fallan, el pipeline se detiene.

Suscrito a: model.trained
Publica:    qa.passed | qa.failed
"""

from typing import Any, Dict

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class QAAgent(AgentBase):
    """Agente de calidad: valida el modelo entrenado contra quality gates.

    Evalúa: métricas primarias contra umbrales configurados en quality_gates
    del YAML. Extensible: puede ejecutar pytest externamente o evaluar
    métricas directamente.
    """

    def __init__(self, bus: EventBus):
        super().__init__(name="qa_engineer", role="QAE", bus=bus)
        self._model_to_validate: Dict[str, Any] = {}
        self._validation_pending = False
        self.subscribe(Topics.MODEL_TRAINED)

    def on_message(self, message):
        super().on_message(message)
        if message.topic == Topics.MODEL_TRAINED:
            self._model_to_validate = message.payload
            self._validation_pending = True

    def observe(self) -> Observation:
        return Observation(data={
            "validation_pending": self._validation_pending,
            "model_info": self._model_to_validate,
        })

    def decide(self, observation: Observation) -> Decision:
        if not observation.data["validation_pending"]:
            return Decision(action="noop", reason="No model to validate")

        return Decision(
            action="validate",
            reason="New model awaiting QA gate",
            params=observation.data["model_info"],
        )

    def act(self, decision: Decision) -> ActionResult:
        """Valida métricas del modelo contra quality gates del config."""
        self._validation_pending = False
        model_info = decision.params

        try:
            from mlops_common import load_config
            cfg = load_config()
            gates = cfg.quality_gates
            metrics = model_info.get("metrics", {})
            failures = []

            for gate_name, threshold in gates.items():
                metric_name = gate_name.replace("min_", "").replace("max_", "")
                metric_value = metrics.get(metric_name)

                if metric_value is None:
                    continue

                if gate_name.startswith("min_") and metric_value < threshold:
                    failures.append(
                        f"{metric_name}={metric_value:.4f} < {threshold} (min)"
                    )
                elif gate_name.startswith("max_") and metric_value > threshold:
                    failures.append(
                        f"{metric_name}={metric_value:.4f} > {threshold} (max)"
                    )

            if failures:
                self.logger.warning("QA FAILED: %s", failures)
                self.publish(Topics.QA_FAILED, {
                    "run_id": model_info.get("run_id"),
                    "failures": failures,
                    "metrics": metrics,
                })
                return ActionResult(
                    success=False,
                    detail=f"QA gate failed: {failures}",
                    artifacts={"failures": failures},
                )

            self.logger.info("QA PASSED — all gates satisfied")
            self.publish(Topics.QA_PASSED, {
                "run_id": model_info.get("run_id"),
                "metrics": metrics,
                "primary_metric": model_info.get("primary_metric"),
                "primary_value": model_info.get("primary_value"),
                "config_name": model_info.get("config_name"),
            })
            return ActionResult(
                success=True,
                detail="All quality gates passed",
                artifacts={"metrics": metrics},
            )

        except Exception as e:
            self.logger.exception("QA validation error")
            self.publish(Topics.QA_FAILED, {
                "run_id": model_info.get("run_id"),
                "error": str(e),
            })
            return ActionResult(success=False, detail=f"QA error: {e}")
