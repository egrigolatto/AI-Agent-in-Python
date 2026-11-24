"""Eventos que emite el núcleo del agente.

Por qué existe este módulo: el agente no imprime por pantalla ni hace
preguntas directamente. Cada paso relevante del ciclo se comunica como una
dataclass inmutable (RF-11) que las interfaces (CLI, TUI, Web) renderizan a
su manera. Esta separación es la que permite tener tres interfaces sobre el
mismo núcleo sin duplicar lógica (D-05).

Convención: los nombres están en pasado (`ToolExecuted`, no `ExecuteTool`);
son notificaciones de algo que ya ocurrió o que necesita una respuesta vía
`confirm_handler` (el único punto bidireccional del ciclo).
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    """Clase base de la jerarquía de eventos del agente."""


@dataclass(frozen=True)
class RunStarted(AgentEvent):
    """El agente comenzó a procesar un turno del usuario."""

    user_input: str


@dataclass(frozen=True)
class LLMResponseReceived(AgentEvent):
    """El proveedor devolvió una respuesta (puede traer tool calls o texto)."""

    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    iteration: int


@dataclass(frozen=True)
class ToolCallReceived(AgentEvent):
    """El modelo solicitó ejecutar una herramienta.

    `arguments` es None cuando el JSON venía malformado; en ese caso
    `is_error` es True y `error_detail` trae el mensaje que se devolverá al
    modelo para que reintente (RF-12).
    """

    tool_call_id: str
    tool_name: str
    raw_arguments: str
    arguments: dict[str, Any] | None
    requires_confirmation: bool
    is_error: bool = False
    error_detail: str | None = None


@dataclass(frozen=True)
class ConfirmationRequested(AgentEvent):
    """La herramienta es peligrosa y el agente va a pedir aprobación (D-03)."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ConfirmationResolved(AgentEvent):
    """El usuario (o el handler de la interfaz) aprobó o rechazó la acción."""

    tool_call_id: str
    tool_name: str
    approved: bool


@dataclass(frozen=True)
class ToolExecuted(AgentEvent):
    """Resultado final de un tool call (ejecutado, denegado o con error)."""

    tool_call_id: str
    tool_name: str
    result: str
    is_error: bool
    denied: bool = False


@dataclass(frozen=True)
class MessageDelta(AgentEvent):
    """Fragmento de texto generado por el modelo durante el streaming.

    Solo se emite cuando el agente corre con `stream=True` y el proveedor
    implementa `stream_complete`; las interfaces lo usan para mostrar la
    respuesta token a token (TUI/Web).
    """

    text: str


@dataclass(frozen=True)
class FinalMessage(AgentEvent):
    """El modelo produjo una respuesta de texto final (sin tool calls)."""

    content: str


@dataclass(frozen=True)
class MaxIterationsReached(AgentEvent):
    """El ciclo agotó `max_iterations` sin respuesta final (protección RNF-02)."""

    max_iterations: int
