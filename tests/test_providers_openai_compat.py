"""Pruebas del adaptador OpenAI-compatible con un cliente stub.

Qué cubren: la traducción de tipos propios ↔ formato de la API en ambos
sentidos (`message_to_openai` y el parseo de `complete()`), el manejo de
respuestas sin `usage` (algunos servidores no lo reportan), sin `choices`
(error explícito) y la AGREGACIÓN del streaming: tool calls fragmentados por
`index` que deben reconstruirse completos en `StreamDone`.

No se toca la red: se inyecta un cliente falso con la forma mínima que usa el
adaptador (`client.chat.completions.create`), gracias a que
`OpenAICompatibleProvider` acepta `client=`.
"""

from types import SimpleNamespace

import pytest

from ai_agent.providers.base import ChatMessage, TextDelta, ToolCall
from ai_agent.providers.openai_compat import OpenAICompatibleProvider, message_to_openai


def make_message(role, **kwargs):
    return ChatMessage(role=role, **kwargs)


class TestMessageToOpenAI:
    def test_system_y_user(self):
        assert message_to_openai(make_message("system", content="instrucciones")) == {
            "role": "system",
            "content": "instrucciones",
        }
        assert message_to_openai(make_message("user", content="hola")) == {
            "role": "user",
            "content": "hola",
        }

    def test_asistente_sin_content_con_tool_calls(self):
        message = make_message(
            "assistant",
            content=None,
            tool_calls=[ToolCall(id="c1", name="say", arguments='{"text": "x"}')],
        )
        payload = message_to_openai(message)
        assert payload["content"] is None
        assert payload["tool_calls"] == [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "say", "arguments": '{"text": "x"}'},
            }
        ]

    def test_mensaje_de_herramienta_incluye_tool_call_id(self):
        message = make_message("tool", content="resultado", tool_call_id="c9")
        payload = message_to_openai(message)
        assert payload == {"role": "tool", "content": "resultado", "tool_call_id": "c9"}

    def test_tool_sin_content_no_produce_none(self):
        payload = message_to_openai(make_message("tool", content=None, tool_call_id="c1"))
        assert payload["content"] == ""  # la API rechaza content=None en tool


def response_stub(*, content, tool_calls=None, usage=True, model="stub-model"):
    mensaje = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in (tool_calls or [])
        ],
    )
    return SimpleNamespace(
        model=model,
        choices=[SimpleNamespace(message=mensaje, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7) if usage else None,
    )


class StubCompletions:
    def __init__(self, result) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class StubClient:
    def __init__(self, result) -> None:
        self.chat = SimpleNamespace(completions=StubCompletions(result))
        self.closed = False

    async def close(self):
        self.closed = True


def make_provider(result):
    client = StubClient(result)
    provider = OpenAICompatibleProvider(
        base_url="http://stub.test/v1", api_key="k", model="stub", client=client
    )
    return provider, client


class TestComplete:
    async def test_traduce_respuesta_de_texto(self):
        provider, _ = make_provider(response_stub(content="hola"))

        completion = await provider.complete([make_message("user", content="x")])

        assert completion.message.role == "assistant"
        assert completion.message.content == "hola"
        assert completion.usage.prompt_tokens == 11
        assert completion.model == "stub-model"

    async def test_traduce_tool_calls(self):
        provider, _ = make_provider(
            response_stub(
                content=None,
                tool_calls=[{"id": "c1", "name": "say", "arguments": '{"a": 1}'}],
            )
        )

        completion = await provider.complete([make_message("user", content="x")])

        assert completion.message.tool_calls == [
            ToolCall(id="c1", name="say", arguments='{"a": 1}')
        ]

    async def test_sin_usage_no_rompe(self):
        provider, _ = make_provider(response_stub(content="ok", usage=False))
        completion = await provider.complete([make_message("user", content="x")])
        assert completion.usage is None

    async def test_envia_esquemas_cuando_hay_herramientas(self):
        provider, client = make_provider(response_stub(content="ok"))
        await provider.complete(
            [make_message("user", content="x")], tools=[{"type": "function"}]
        )
        assert client.chat.completions.calls[0]["tools"] == [{"type": "function"}]

    async def test_choices_vacio_es_error_explicito(self):
        sin_choices = SimpleNamespace(model="m", choices=[], usage=None)
        provider, _ = make_provider(sin_choices)
        with pytest.raises(RuntimeError, match="sin choices"):
            await provider.complete([make_message("user", content="x")])

    async def test_aclose_cierra_cliente(self):
        provider, client = make_provider(response_stub(content="ok"))
        await provider.aclose()
        assert client.closed is True


class FakeStream:
    """Respuesta iterable estilo `AsyncOpenAI` para pruebas de streaming."""

    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)

    def __aiter__(self):
        async def generar():
            for chunk in self._chunks:
                yield chunk

        return generar()


def chunk(text=None, tool_call=None, *, model=None, usage=None, finish=None):
    delta = SimpleNamespace(
        content=text,
        tool_calls=(
            [
                SimpleNamespace(
                    index=tool_call["index"],
                    id=tool_call.get("id"),
                    function=SimpleNamespace(
                        name=tool_call.get("name"), arguments=tool_call.get("arguments")
                    ),
                )
            ]
            if tool_call
            else None
        ),
    )
    # Chunk sin choices = el chunk final que solo trae usage (algunos servidores
    # lo emiten así); si trae `finish`, se conserva su choice para el motivo.
    choices = (
        []
        if (usage is not None and tool_call is None and text is None and finish is None)
        else [SimpleNamespace(delta=delta, finish_reason=finish)]
    )
    return SimpleNamespace(
        choices=choices,
        model=model,
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3) if usage else None,
    )


class TestStreamComplete:
    async def test_agrega_texto_y_emite_deltas(self):
        provider, _ = make_provider(
            FakeStream([chunk(text="Ho"), chunk(text="la"), chunk(usage=True)])
        )
        events = [
            event
            async for event in provider.stream_complete([make_message("user", content="x")])
        ]
        deltas = [event for event in events if isinstance(event, TextDelta)]
        assert "".join(d.text for d in deltas) == "Hola"
        done = events[-1]
        assert done.completion.message.content == "Hola"
        assert done.completion.usage.prompt_tokens == 5

    async def test_reconstruye_tool_call_fragmentado(self):
        # La API fragmenta el tool call: id+nombre primero, argumentos en trozos.
        chunks = [
            chunk(tool_call={"index": 0, "id": "c1", "name": "write_file", "arguments": '{"file_'}),
            chunk(tool_call={"index": 0, "arguments": 'path": "a.txt", "content": "h'}),
            chunk(tool_call={"index": 0, "arguments": 'i"}'}),
            chunk(usage=True, finish="tool_calls"),
        ]
        provider, _ = make_provider(FakeStream(chunks))

        events = [
            event
            async for event in provider.stream_complete([make_message("user", content="x")])
        ]

        tool_calls = events[-1].completion.message.tool_calls
        assert tool_calls == [
            ToolCall(
                id="c1",
                name="write_file",
                arguments='{"file_path": "a.txt", "content": "hi"}',
            )
        ]
        assert events[-1].completion.finish_reason == "tool_calls"

    async def test_tool_call_sin_id_recibe_id_sintetico(self):
        chunks = [
            chunk(tool_call={"index": 0, "name": "say", "arguments": "{}"}),
            chunk(usage=True),
        ]
        provider, _ = make_provider(FakeStream(chunks))
        events = [
            event
            async for event in provider.stream_complete([make_message("user", content="x")])
        ]
        assert events[-1].completion.message.tool_calls[0].id == "call_0"
