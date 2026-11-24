"""Núcleo del agente: el bucle de tool calling.

Resumen conceptual (detalle en `docs/technical-design.md`): un "agente" es un
LLM que, en lugar de limitarse a generar texto, puede pedir la ejecución de
herramientas y usar sus resultados para seguir razonando. El bucle es::

    mensaje usuario → LLM → (texto final | tool calls) → resultados → LLM → ...

Esta clase implementa ese bucle y nada más: mantiene el historial
(multi-turno, RF-03), delega la ejecución en el `ToolRegistry` con
confirmación humana para las herramientas peligrosas (RF-05), tolera tool
calls malformados devolviendo el error al modelo (RF-12) y notifica cada
paso con eventos (RF-11). No imprime por pantalla ni lee del teclado: de eso
se ocupan las interfaces (CLI, TUI, Web), que solo consumen eventos y proveen
un `confirm_handler`.

Decisiones de diseño (D-05):

- async: textual y FastAPI son async-nativos; las herramientas bloqueantes
  (archivo/subprocess) se ejecutan con `asyncio.to_thread` (RNF-05).
- Callbacks en vez de generators bidireccionales: más simple de seguir para
  un proyecto de aprendizaje; el único punto bidireccional es
  `confirm_handler`.
"""

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_agent import events
from ai_agent.events import AgentEvent
from ai_agent.prompts import SYSTEM_PROMPT
from ai_agent.providers.base import (
    ChatMessage,
    Completion,
    LLMProvider,
    StreamDone,
    TextDelta,
    ToolCall,
)
from ai_agent.tools.registry import ToolContext, ToolRegistry

EventCallback = Callable[[AgentEvent], Awaitable[None] | None]


@dataclass(frozen=True)
class ConfirmationRequest:
    """Información que recibe la interfaz para pedir la aprobación al usuario."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


ConfirmationHandler = Callable[[ConfirmationRequest], Awaitable[bool] | bool]


class Agent:
    """Agente con historial persistente entre turnos y ciclo de herramientas.

    Uso mínimo::

        agent = Agent(provider=..., tools=build_default_registry(), workdir="./calculator")
        respuesta = await agent.run("Lista los archivos del proyecto")
    """

    def __init__(
        self,
        *,
        provider: LLMProvider,
        tools: ToolRegistry,
        workdir: str | Path,
        system_prompt: str = SYSTEM_PROMPT,
        max_iterations: int = 20,
        max_file_chars: int = 10_000,
        stream: bool = False,
        on_event: EventCallback | None = None,
        confirm_handler: ConfirmationHandler | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_iterations = max_iterations
        self.stream = stream
        # Ruta absoluta: el sandbox no debe depender del CWD del proceso (RF-06).
        self.context = ToolContext(
            working_directory=str(Path(workdir).resolve()),
            max_file_chars=max_file_chars,
        )
        self._on_event = on_event
        self._confirm_handler = confirm_handler
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]

    # --- API pública -------------------------------------------------------

    async def run(self, user_input: str) -> str:
        """Procesa un turno del usuario y devuelve la respuesta final.

        El historial se conserva entre llamadas a `run()`: eso habilita la
        conversación multi-turno sin estado externo (RF-03).
        """
        self.history.append(ChatMessage(role="user", content=user_input))
        await self._emit(events.RunStarted(user_input=user_input))

        for iteration in range(1, self.max_iterations + 1):
            completion = await self._request_completion()
            usage = completion.usage
            await self._emit(
                events.LLMResponseReceived(
                    model=completion.model,
                    prompt_tokens=usage.prompt_tokens if usage else None,
                    completion_tokens=usage.completion_tokens if usage else None,
                    finish_reason=completion.finish_reason,
                    iteration=iteration,
                )
            )

            message = completion.message
            self.history.append(message)

            if not message.tool_calls:
                content = message.content or ""
                await self._emit(events.FinalMessage(content=content))
                return content

            for tool_call in message.tool_calls:
                result = await self._execute_tool_call(tool_call)
                self.history.append(
                    ChatMessage(
                        role="tool", content=result, tool_call_id=tool_call.id
                    )
                )

        # Protección contra bucles infinitos (RNF-02). El historial queda con
        # los últimos resultados de herramientas: si el usuario vuelve a
        # escribir, el modelo puede retomar y concluir.
        await self._emit(events.MaxIterationsReached(max_iterations=self.max_iterations))
        return (
            f"Error: The agent reached the maximum number of iterations "
            f"({self.max_iterations}) without a final response."
        )

    def reset(self) -> None:
        """Vacía el historial conservando el mensaje del sistema."""
        self.history = [self.history[0]]

    def set_callbacks(
        self,
        *,
        on_event: EventCallback | None = None,
        confirm_handler: ConfirmationHandler | None = None,
    ) -> None:
        """Reconfigura los callbacks del agente en caliente.

        Por qué existe: cada interfaz los gestiona distinto. La CLI/TUI los fijan en la
        construcción (una sola sesión por proceso); la Web necesita un emisor de eventos y
        un confirmador DIFERENTES en cada turno (una cola SSE y futures de confirmación
        por request). Solo es seguro porque el registro impone un turno activo por sesión
        (la API Web devuelve 409 si el agente está ocupado).
        """
        self._on_event = on_event
        self._confirm_handler = confirm_handler

    # --- Internos -----------------------------------------------------------

    async def _request_completion(self) -> Completion:
        """Pide una respuesta al proveedor, con streaming si está habilitado.

        Si `stream=True` y el proveedor implementa `stream_complete`, consume
        el stream reemitiendo cada fragmento de texto como evento
        `MessageDelta` (para que la interfaz muestre tokens en vivo). El
        resultado agregado (`StreamDone.completion`) es idéntico al que
        devolvería `complete()`, así que el resto del bucle no distingue
        entre ambos modos. Si el proveedor no soporta streaming, se usa
        `complete()` directamente: la capacidad es opcional (ver
        `providers/base.py`).
        """
        tools = self.tools.schemas() or None
        stream_complete = getattr(self.provider, "stream_complete", None)
        if not (self.stream and stream_complete is not None):
            return await self.provider.complete(self.history, tools)

        completion: Completion | None = None
        async for stream_event in stream_complete(self.history, tools):
            if isinstance(stream_event, TextDelta):
                await self._emit(events.MessageDelta(text=stream_event.text))
            elif isinstance(stream_event, StreamDone):
                completion = stream_event.completion
        if completion is None:
            raise RuntimeError(
                "El stream del proveedor terminó sin emitir un evento StreamDone."
            )
        return completion

    async def _emit(self, event: AgentEvent) -> None:
        """Notifica un evento a la interfaz (callback sync o async)."""
        if self._on_event is None:
            return
        result = self._on_event(event)
        if inspect.isawaitable(result):
            await result

    async def _confirm(self, request: ConfirmationRequest) -> bool:
        """Consulta al usuario vía interfaz. Sin handler, se aprueba (headless)."""
        if self._confirm_handler is None:
            return True
        result = self._confirm_handler(request)
        if inspect.isawaitable(result):
            return bool(await result)
        return bool(result)

    async def _execute_tool_call(self, tool_call: ToolCall) -> str:
        """Ejecuta un tool call completo: parseo, confirmación, ejecución, evento.

        Devuelve el texto que se agregará al historial como mensaje `tool`.
        Nunca lanza excepciones por causas del modelo (JSON inválido,
        herramienta desconocida): todo se convierte en un resultado de error
        que el modelo recibe para autocorregirse (RF-12).
        """
        parsed_arguments, parse_error = self._parse_arguments(tool_call)
        tool = self.tools.get(tool_call.name)
        requires_confirmation = bool(tool is not None and tool.requires_confirmation)

        await self._emit(
            events.ToolCallReceived(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                raw_arguments=tool_call.arguments,
                arguments=parsed_arguments,
                requires_confirmation=requires_confirmation,
                is_error=parse_error is not None,
                error_detail=parse_error,
            )
        )

        denied = False
        if parse_error is not None:
            result = parse_error
        else:
            if requires_confirmation:
                request = ConfirmationRequest(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    arguments=parsed_arguments or {},
                )
                await self._emit(
                    events.ConfirmationRequested(
                        tool_call_id=request.tool_call_id,
                        tool_name=request.tool_name,
                        arguments=request.arguments,
                    )
                )
                approved = await self._confirm(request)
                await self._emit(
                    events.ConfirmationResolved(
                        tool_call_id=request.tool_call_id,
                        tool_name=request.tool_name,
                        approved=approved,
                    )
                )
                if not approved:
                    denied = True
                    result = (
                        f"Denied: The user denied the execution of "
                        f"'{tool_call.name}'. Do not retry this action; "
                        "continue your response acknowledging the denial."
                    )
            if not denied:
                # Los handlers pueden bloquear (I/O de archivo, subprocess):
                # se ejecutan fuera del event loop (RNF-05).
                result = await asyncio.to_thread(
                    self.tools.call, tool_call.name, parsed_arguments or {}, self.context
                )

        await self._emit(
            events.ToolExecuted(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                result=result,
                is_error=parse_error is not None or result.startswith("Error:"),
                denied=denied,
            )
        )
        return result

    def _parse_arguments(self, tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
        """Parsea el JSON de argumentos sin romper el ciclo.

        Devuelve `(arguments, None)` si el JSON es un objeto válido, o
        `(None, mensaje_para_el_modelo)` si está malformado. El mensaje
        incluye el error y el texto recibido para ayudar al reintento, un
        fallo común en modelos locales pequeños (RF-12).
        """
        raw = tool_call.arguments or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            return None, (
                f"Error: Invalid JSON arguments for tool '{tool_call.name}': "
                f"{error}. Received: {raw!r}. Retry with valid JSON."
            )
        if not isinstance(parsed, dict):
            return None, (
                f"Error: Arguments for tool '{tool_call.name}' must be a JSON "
                f"object; received {type(parsed).__name__}. Retry with valid JSON."
            )
        return parsed, None
