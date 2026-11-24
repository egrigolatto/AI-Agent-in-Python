"""Abstracción de proveedores: tipos comunes e interfaz.

Por qué existe este módulo: el núcleo del agente (`agent.py`) no debe saber
con qué API habla. Aquí se definen tipos de mensaje propios
(`ChatMessage`, `ToolCall`) y el `Protocol LLMProvider`; cada adaptador
(p. ej. `openai_compat.py`) traduce entre estos tipos y los de su SDK.
Así, agregar un proveedor nuevo no toca la lógica del agente (RF-01, RNF-06).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    """Solicitud del modelo para ejecutar una herramienta.

    `arguments` se conserva como el string JSON crudo que generó el modelo.
    El parseo ocurre en el agente (no aquí) para poder tolerar JSON inválido
    sin perder la información original (RF-12): los modelos locales a veces
    producen argumentos malformados y el reintento necesita ver qué envió.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatMessage:
    """Mensaje de la conversación en formato independiente del proveedor.

    Roles: `system` (instrucciones permanentes), `user` (entrada humana),
    `assistant` (respuesta del modelo, puede traer `tool_calls`), `tool`
    (resultado de una herramienta, identificado por `tool_call_id`).
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Usage:
    """Consumo de tokens reportado por el proveedor (puede no existir)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class Completion:
    """Respuesta completa del proveedor a una solicitud del agente."""

    message: ChatMessage
    usage: Usage | None = None
    model: str | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """Fragmento incremental de texto del asistente durante el streaming."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """Fin del streaming: trae la respuesta completa agregada.

    El agente usa este `Completion` exactamente igual que el de `complete()`:
    el streaming solo cambia cómo se *muestra* la generación, no la lógica
    del ciclo (los tool calls llegan agregados al final).
    """

    completion: Completion


StreamEvent = TextDelta | StreamDone

ToolSchema = dict[str, Any]


@runtime_checkable
class LLMProvider(Protocol):
    """Interfaz mínima que debe cumplir cualquier adaptador de proveedor.

    Un proveedor necesita dos operaciones: completar una conversación
    (`complete`) y liberar recursos (`aclose`).

    Opcionalmente puede implementar `stream_complete` (P-04 resuelto en la
    Fase 3): un generador asíncrono que emite `TextDelta` mientras el modelo
    genera y termina con `StreamDone`. El agente lo usa solo si se construyó
    con `stream=True` y el proveedor lo ofrece; si no, cae a `complete()`.
    Esto mantiene los proveedores simples (y los tests con `FakeProvider`)
    sin obligarlos a implementar streaming.
    """

    model: str

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> Completion:
        """Envía el historial (y los esquemas de herramientas) y devuelve una respuesta."""
        ...

    async def aclose(self) -> None:
        """Libera los recursos HTTP subyacentes."""
        ...
