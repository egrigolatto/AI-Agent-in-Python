"""Adaptador para APIs compatibles con la API de OpenAI.

Qué resuelve: casi todos los proveedores actuales (OpenRouter, Ollama en
`http://localhost:11434/v1`, OpenAI, Groq, Together, DeepInfra, ...) exponen
la misma API de facto inventada por OpenAI (`/chat/completions` con
`tools`). En lugar de escribir un adaptador por proveedor, uno solo cubre
todos: cambian únicamente `base_url`, `api_key` y `model` (decisión D-01).

Cómo funciona: recibe los tipos propios del núcleo (`ChatMessage`), los
traduce a los dicts que espera el SDK (`message_to_openai`), llama a
`AsyncOpenAI` y traduce la respuesta de vuelta a `Completion`. El núcleo
nunca ve tipos del SDK (RNF-06).

Streaming: `stream_complete` consume el modo `stream=True` de la API, que
devuelve "chunks" con deltas parciales. Los deltas de texto se reemiten como
`TextDelta` (para que la interfaz muestre tokens en vivo) y los deltas de
tool calls —que llegan fragmentados e indexados— se acumulan hasta
reconstruir el `ToolCall` completo que se entrega en `StreamDone`.
"""

from collections.abc import AsyncIterator, Sequence

from openai import AsyncOpenAI

from ai_agent.providers.base import (
    ChatMessage,
    Completion,
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolSchema,
    Usage,
)


def message_to_openai(message: ChatMessage) -> dict:
    """Convierte un mensaje del núcleo al formato dict de la API de OpenAI.

    Detalles importantes del protocolo:
    - Un `assistant` puede tener `content=None` cuando solo pide herramientas.
    - Los `tool_calls` viajan como dicts anidados con `type="function"`.
    - Un mensaje `role="tool"` debe llevar `tool_call_id` para que la API
      relacione el resultado con la solicitud original, y `content` no nulo.
    """
    payload: dict = {"role": message.role}
    if message.role == "assistant":
        payload["content"] = message.content
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
    else:
        payload["content"] = message.content if message.content is not None else ""
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id
    return payload


class OpenAICompatibleProvider:
    """Proveedor LLM sobre `AsyncOpenAI`, configurable por `base_url`.

    Cumple el `Protocol LLMProvider` de `providers/base.py`.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 120.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        # `client` inyectable: los tests pasan un sustituto sin tocar la red.
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_request_kwargs(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None,
    ) -> dict:
        """Arma el payload común de `/chat/completions` para ambos modos."""
        request_kwargs: dict = {
            "model": self.model,
            "messages": [message_to_openai(message) for message in messages],
            "temperature": self.temperature,
        }
        if tools:
            request_kwargs["tools"] = list(tools)
        return request_kwargs

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> Completion:
        response = await self._client.chat.completions.create(
            **self._build_request_kwargs(messages, tools)
        )
        if not response.choices:
            raise RuntimeError(
                f"El proveedor '{self.base_url}' devolvió una respuesta sin choices."
            )

        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=raw_call.function.arguments or "{}",
            )
            for raw_call in (message.tool_calls or [])
        ]
        return Completion(
            message=ChatMessage(
                role="assistant",
                content=message.content,
                tool_calls=tool_calls or None,
            ),
            usage=_to_usage(getattr(response, "usage", None)),
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def stream_complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Versión streaming de `complete()`.

        Cómo se agregan los chunks: la API envía deltas numerados por
        `index` (un tool call puede llegar en decenas de fragmentos: primero
        el nombre, luego trozos del JSON de argumentos). Se acumulan en un
        dict por índice y al cerrar el stream se reconstruyen los `ToolCall`
        completos. El texto se reemite en el momento (`TextDelta`) para que
        las interfaces muestren la generación token a token.

        Nota: los tokens de razonamiento de los modelos "thinking" (campo
        `reasoning_content` en Ollama) no se reemiten como texto: solo se
        muestra el contenido final de la respuesta.
        """
        request_kwargs = self._build_request_kwargs(messages, tools)
        request_kwargs["stream"] = True
        # Pide el consumo de tokens en el último chunk; algunos servidores
        # compatibles lo ignoran, por eso `usage` puede quedar en None.
        request_kwargs["stream_options"] = {"include_usage": True}

        stream = await self._client.chat.completions.create(**request_kwargs)

        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict] = {}
        usage: Usage | None = None
        model: str | None = None
        finish_reason: str | None = None

        async for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = _to_usage(chunk.usage)
            if getattr(chunk, "model", None):
                model = chunk.model
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                content_parts.append(delta.content)
                yield TextDelta(text=delta.content)
            for raw_call in getattr(delta, "tool_calls", None) or []:
                accumulator = tool_calls_by_index.setdefault(
                    raw_call.index, {"id": "", "name": "", "arguments": ""}
                )
                if raw_call.id:
                    accumulator["id"] = raw_call.id
                if raw_call.function:
                    if raw_call.function.name:
                        accumulator["name"] += raw_call.function.name
                    if raw_call.function.arguments:
                        accumulator["arguments"] += raw_call.function.arguments

        tool_calls = [
            ToolCall(
                id=data["id"] or f"call_{index}",
                name=data["name"],
                arguments=data["arguments"] or "{}",
            )
            for index, data in sorted(tool_calls_by_index.items())
            if data["name"]
        ]
        yield StreamDone(
            completion=Completion(
                message=ChatMessage(
                    role="assistant",
                    content="".join(content_parts) or None,
                    tool_calls=tool_calls or None,
                ),
                usage=usage,
                model=model,
                finish_reason=finish_reason,
            )
        )

    async def aclose(self) -> None:
        await self._client.close()


def _to_usage(raw_usage) -> Usage | None:
    """Convierte el objeto `usage` del SDK (si existe) al tipo del núcleo."""
    if raw_usage is None:
        return None
    return Usage(
        prompt_tokens=raw_usage.prompt_tokens or 0,
        completion_tokens=raw_usage.completion_tokens or 0,
    )
