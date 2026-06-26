"""
event_bus.py
────────────
Pub/sub en memoria para comunicación inter-agente.

Implementa el patrón de la matriz de interacción (clase-multiagente-mlops.html,
Parte 04): cada agente publica en tópicos y otros agentes suscritos reaccionan.

Extensible: reemplazar esta clase por un adaptador a Redis Streams, NATS, o
RabbitMQ para agentes distribuidos en múltiples hosts — la interfaz
publish()/subscribe() no cambia.
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, List

from .base import AgentMessage


class EventBus:
    """Canal de comunicación central entre agentes.

    Cada agente publica AgentMessage en tópicos; los suscriptores reciben
    el mensaje sincrónicamente (en memoria) o vía callback.

    Atributos
    ---------
    history : lista ordenada de todos los mensajes publicados (auditoría)
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: List[AgentMessage] = []
        self.logger = logging.getLogger("event_bus")

    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]):
        self._subscribers[topic].append(callback)
        self.logger.debug("Subscribed to '%s'", topic)

    def publish(self, message: AgentMessage):
        self._history.append(message)
        self.logger.info(
            "[%s] → %s: %s",
            message.sender,
            message.topic,
            _summarize(message.payload),
        )
        for cb in self._subscribers.get(message.topic, []):
            try:
                cb(message)
            except Exception:
                self.logger.exception(
                    "Error en handler de '%s' (sender=%s)",
                    message.topic,
                    message.sender,
                )

    @property
    def history(self) -> List[AgentMessage]:
        return list(self._history)

    def clear_history(self):
        self._history.clear()

    def topics_with_subscribers(self) -> Dict[str, int]:
        return {t: len(cbs) for t, cbs in self._subscribers.items() if cbs}


def _summarize(payload: dict, max_len: int = 120) -> str:
    s = str(payload)
    return s if len(s) <= max_len else s[:max_len] + "..."
