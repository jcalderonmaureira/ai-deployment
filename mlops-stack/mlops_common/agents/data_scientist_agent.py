"""
data_scientist_agent.py
───────────────────────
Agente DS (Data Scientist) — Eken et al. §3.3.

Servicio Docker: model-trainer + MLflow.
Responsabilidad: entrenar modelos, registrar en MLflow, evaluar métricas.

Suscrito a: retrain.triggered
Publica:    model.trained
"""

from typing import Any, Dict

from .base import AgentBase, Observation, Decision, ActionResult, Topics
from .event_bus import EventBus


class DataScientistAgent(AgentBase):
    """Agente que entrena y registra modelos vía mlops_common.

    Envuelve la lógica de train.py: carga config → carga datos → construye
    modelo → entrena → evalúa → registra en MLflow → publica resultado.
    """

    def __init__(self, bus: EventBus, mlflow_tracking_uri: str = "http://mlflow:5001"):
        super().__init__(name="data_scientist", role="DS", bus=bus)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self._retrain_requested = False
        self._retrain_params: Dict[str, Any] = {}
        self.subscribe(Topics.RETRAIN_TRIGGERED)

    def on_message(self, message):
        super().on_message(message)
        if message.topic == Topics.RETRAIN_TRIGGERED:
            self._retrain_requested = True
            self._retrain_params = message.payload

    def observe(self) -> Observation:
        """Verifica si hay un request de reentrenamiento pendiente."""
        return Observation(data={
            "retrain_requested": self._retrain_requested,
            "retrain_params": self._retrain_params,
        })

    def decide(self, observation: Observation) -> Decision:
        if not observation.data["retrain_requested"]:
            return Decision(action="noop", reason="No retrain request pending")

        return Decision(
            action="train",
            reason="Retrain requested by orchestrator",
            params=observation.data["retrain_params"],
        )

    def act(self, decision: Decision) -> ActionResult:
        """Entrena un nuevo modelo usando mlops_common y lo registra en MLflow."""
        self._retrain_requested = False
        self._retrain_params = {}

        try:
            import mlflow
            import mlflow.sklearn
            from sklearn.model_selection import train_test_split

            from mlops_common import load_config
            from mlops_common.data_sources import get_data_source
            from mlops_common.models import get_model_strategy
            from mlops_common.problem_types import get_problem_type

            cfg = load_config()
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment_name)

            bundle = get_data_source(cfg.data_source).load()
            strategy = get_model_strategy(cfg.model.family)
            problem_type = get_problem_type(cfg.problem_type)

            split_kwargs = {"test_size": cfg.training.test_size, "random_state": 42}
            if cfg.training.stratify:
                split_kwargs["stratify"] = bundle.y
            X_train, X_test, y_train, y_test = train_test_split(
                bundle.X, bundle.y, **split_kwargs
            )

            model = strategy.build(cfg.model.params)
            strategy.fit(model, X_train, y_train)

            y_pred = strategy.predict(model, X_test)
            y_proba = None
            if strategy.supports_proba:
                y_proba = strategy.predict_proba(model, X_test)

            metrics = problem_type.compute_metrics(y_test, y_pred, y_proba)
            primary_metric = metrics.get(problem_type.primary_metric, 0.0)

            with mlflow.start_run() as run:
                mlflow.log_params(cfg.model.params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, "model")
                run_id = run.info.run_id

            self.logger.info("Model trained: %s = %.4f (run=%s)",
                             problem_type.primary_metric, primary_metric, run_id)

            self.publish(Topics.MODEL_TRAINED, {
                "run_id": run_id,
                "metrics": metrics,
                "primary_metric": problem_type.primary_metric,
                "primary_value": primary_metric,
                "model_family": cfg.model.family,
                "config_name": cfg.name,
            })

            return ActionResult(
                success=True,
                detail=f"Model trained: {problem_type.primary_metric}={primary_metric:.4f}",
                artifacts={"run_id": run_id, "metrics": metrics},
            )

        except Exception as e:
            self.logger.exception("Training failed")
            self.publish(Topics.SYSTEM_ERROR, {
                "agent": self.name,
                "error": str(e),
            })
            return ActionResult(success=False, detail=f"Training failed: {e}")
