"""Pruebas del núcleo del agente con un proveedor falso (sin red ni claves).

Qué verifican: el bucle de tool calling completo (RF-03), la confirmación
human-in-the-loop aprobada y rechazada (RF-05), la tolerancia a tool calls
malformados (RF-12), el límite de iteraciones (RNF-02), la emisión de eventos
(RF-11) y el historial multi-turno. El `FakeProvider` devuelve respuestas
guionizadas, lo que hace los tests deterministas y ejecutables en cualquier
máquina (RF-13).
"""

from pathlib import Path

from ai_agent.agent import Agent
from ai_agent.events import (
    AgentEvent,
    ConfirmationResolved,
    FinalMessage,
    LLMResponseReceived,
    MaxIterationsReached,
    MessageDelta,
    RunStarted,
    ToolCallReceived,
    ToolExecuted,
)
from ai_agent.providers.base import (
    ChatMessage,
    Completion,
    StreamDone,
    TextDelta,
    ToolCall,
    Usage,
)
from ai_agent.tools.registry import Tool, ToolRegistry


class FakeProvider:
    """Proveedor guionizado: cumple el Protocol LLMProvider sin tocar la red."""

    model = "fake-model"

    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)
        self.requests: list[dict] = []

    async def complete(self, messages, tools=None) -> Completion:
        self.requests.append({"messages": list(messages), "tools": tools})
        assert self._completions, "FakeProvider se quedó sin respuestas guionizadas"
        return self._completions.pop(0)

    async def aclose(self) -> None:
        pass


class EventCollector:
    """Callback `on_event` que guarda los eventos para inspeccionarlos."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type):
        return [event for event in self.events if isinstance(event, event_type)]


def text_completion(text: str) -> Completion:
    return Completion(
        message=ChatMessage(role="assistant", content=text),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        model="fake-model",
        finish_reason="stop",
    )


def tool_completion(*calls: ToolCall) -> Completion:
    return Completion(
        message=ChatMessage(role="assistant", content=None, tool_calls=list(calls)),
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        model="fake-model",
        finish_reason="tool_calls",
    )


def make_registry() -> ToolRegistry:
    def say(text: str) -> str:
        return f"said: {text}"

    def append_file(working_directory: str, file_path: str, content: str) -> str:
        path = Path(working_directory) / file_path
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(previous + content, encoding="utf-8")
        return f"appended to {file_path}"

    return ToolRegistry(
        [
            Tool(
                name="say",
                description="Repite un texto",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
                handler=say,
            ),
            Tool(
                name="append_file",
                description="Agrega texto a un archivo (peligrosa: requiere confirmación)",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
                handler=append_file,
                requires_confirmation=True,
            ),
        ]
    )


def make_agent(tmp_path: Path, provider: FakeProvider, **kwargs) -> Agent:
    return Agent(
        provider=provider,
        tools=make_registry(),
        workdir=tmp_path,
        **kwargs,
    )


SAY_CALL = ToolCall(id="c1", name="say", arguments='{"text": "hola"}')
APPEND_CALL = ToolCall(
    id="c2", name="append_file", arguments='{"file_path": "f.txt", "content": "X"}'
)


class TestCicloBasico:
    async def test_respuesta_de_texto_simple(self, tmp_path):
        provider = FakeProvider([text_completion("Hola!")])

        result = await make_agent(tmp_path, provider).run("saluda")

        assert result == "Hola!"
        # system + user + assistant
        assert len(provider.requests[0]["messages"]) == 2
        assert provider.requests[0]["messages"][0].role == "system"

    async def test_envia_esquemas_de_herramientas(self, tmp_path):
        provider = FakeProvider([text_completion("ok")])

        await make_agent(tmp_path, provider).run("x")

        tool_names = {
            schema["function"]["name"] for schema in provider.requests[0]["tools"]
        }
        assert tool_names == {"say", "append_file"}

    async def test_ciclo_con_herramienta_devuelve_resultado_al_modelo(self, tmp_path):
        provider = FakeProvider(
            [tool_completion(SAY_CALL), text_completion("listo")]
        )

        result = await make_agent(tmp_path, provider).run("di hola")

        assert result == "listo"
        ultimo_mensaje_segunda_llamada = provider.requests[1]["messages"][-1]
        assert ultimo_mensaje_segunda_llamada == ChatMessage(
            role="tool", content="said: hola", tool_call_id="c1"
        )

    async def test_orden_de_eventos_del_ciclo(self, tmp_path):
        provider = FakeProvider([tool_completion(SAY_CALL), text_completion("fin")])
        collector = EventCollector()

        await make_agent(tmp_path, provider, on_event=collector).run("x")

        # Orden esperado del ciclo:
        # RunStarted → LLMResponseReceived → ToolCallReceived → ToolExecuted
        # → LLMResponseReceived → FinalMessage
        tipos = [type(event) for event in collector.events]
        assert tipos == [
            RunStarted,
            LLMResponseReceived,
            ToolCallReceived,
            ToolExecuted,
            LLMResponseReceived,
            FinalMessage,
        ]


class TestConfirmacion:
    async def tool_message(self, tmp_path, confirm_handler):
        provider = FakeProvider(
            [tool_completion(APPEND_CALL), text_completion("ok")]
        )
        await make_agent(tmp_path, provider, confirm_handler=confirm_handler).run("x")
        return provider, provider.requests[1]["messages"][-1]

    async def test_denegada_no_ejecuta_y_avisa_al_modelo(self, tmp_path):
        provider, tool_message = await self.tool_message(tmp_path, lambda request: False)

        assert tool_message.content.startswith("Denied:")
        assert not (tmp_path / "f.txt").exists()

    async def test_aprobada_ejecuta(self, tmp_path):
        provider, tool_message = await self.tool_message(tmp_path, lambda request: True)

        assert tool_message.content == "appended to f.txt"
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "X"

    async def test_handler_recibe_nombre_y_argumentos(self, tmp_path):
        recibidas = []

        def handler(request):
            recibidas.append(request)
            return False

        await self.tool_message(tmp_path, handler)

        assert recibidas[0].tool_name == "append_file"
        assert recibidas[0].arguments == {"file_path": "f.txt", "content": "X"}

    async def test_handler_async(self, tmp_path):
        async def handler(request):
            return True

        provider, tool_message = await self.tool_message(tmp_path, handler)
        assert tool_message.content == "appended to f.txt"

    async def test_sin_handler_se_autoriza(self, tmp_path):
        provider = FakeProvider([tool_completion(APPEND_CALL), text_completion("ok")])
        await make_agent(tmp_path, provider).run("x")
        assert (tmp_path / "f.txt").exists()

    async def test_eventos_de_confirmacion(self, tmp_path):
        provider = FakeProvider([tool_completion(APPEND_CALL), text_completion("ok")])
        collector = EventCollector()

        await make_agent(
            tmp_path, provider, on_event=collector, confirm_handler=lambda r: False
        ).run("x")

        resolved = collector.of_type(ConfirmationResolved)
        assert len(resolved) == 1 and resolved[0].approved is False
        executed = collector.of_type(ToolExecuted)
        assert executed[0].denied is True


class TestToleranciaAFallos:
    async def test_json_malformado_vuelve_como_error_al_modelo(self, tmp_path):
        malformed = ToolCall(id="c1", name="say", arguments='{"text": ')
        provider = FakeProvider([tool_completion(malformed), text_completion("ok")])

        await make_agent(tmp_path, provider).run("x")

        tool_message = provider.requests[1]["messages"][-1]
        assert "Invalid JSON" in tool_message.content
        assert tool_message.tool_call_id == "c1"

    async def test_argumentos_no_objeto(self, tmp_path):
        weird = ToolCall(id="c1", name="say", arguments='["no", "objeto"]')
        provider = FakeProvider([tool_completion(weird), text_completion("ok")])

        await make_agent(tmp_path, provider).run("x")

        assert "must be a JSON object" in provider.requests[1]["messages"][-1].content

    async def test_herramienta_desconocida(self, tmp_path):
        ghost = ToolCall(id="c1", name="no_existe", arguments="{}")
        provider = FakeProvider([tool_completion(ghost), text_completion("ok")])

        await make_agent(tmp_path, provider).run("x")

        assert "Unknown tool" in provider.requests[1]["messages"][-1].content

    async def test_error_de_herramienta_no_rompe_el_ciclo(self, tmp_path):
        bad_args = ToolCall(id="c1", name="say", arguments="{}")  # falta 'text'
        provider = FakeProvider([tool_completion(bad_args), text_completion("ok")])

        result = await make_agent(tmp_path, provider).run("x")

        assert result == "ok"
        assert "Error" in provider.requests[1]["messages"][-1].content


class TestLimitesYMultiTurno:
    async def test_limite_de_iteraciones(self, tmp_path):
        provider = FakeProvider([tool_completion(SAY_CALL)] * 3)
        collector = EventCollector()

        result = await make_agent(
            tmp_path, provider, max_iterations=3, on_event=collector
        ).run("x")

        assert result.startswith("Error:")
        assert "maximum number of iterations" in result
        assert len(provider.requests) == 3
        assert len(collector.of_type(MaxIterationsReached)) == 1

    async def test_historial_multi_turno(self, tmp_path):
        provider = FakeProvider([text_completion("uno"), text_completion("dos")])
        agent = make_agent(tmp_path, provider)

        assert await agent.run("primer turno") == "uno"
        assert await agent.run("segundo turno") == "dos"

        mensajes_segunda_llamada = provider.requests[1]["messages"]
        contenidos = [m.content for m in mensajes_segunda_llamada]
        assert "primer turno" in contenidos
        assert "uno" in contenidos
        assert "segundo turno" in contenidos

    async def test_reset_vacia_el_historial(self, tmp_path):
        provider = FakeProvider([text_completion("uno"), text_completion("dos")])
        agent = make_agent(tmp_path, provider)
        await agent.run("x")

        agent.reset()
        await agent.run("y")

        mensajes_segunda_llamada = provider.requests[1]["messages"]
        assert len(mensajes_segunda_llamada) == 2  # system + user
        assert mensajes_segunda_llamada[0].role == "system"


class FakeStreamingProvider(FakeProvider):
    """FakeProvider con `stream_complete`: fragmenta el texto en deltas de 2 chars."""

    async def stream_complete(self, messages, tools=None):
        self.requests.append({"messages": list(messages), "tools": tools})
        assert self._completions, "FakeProvider se quedó sin respuestas guionizadas"
        completion = self._completions.pop(0)
        content = completion.message.content or ""
        for index in range(0, len(content), 2):
            yield TextDelta(text=content[index : index + 2])
        yield StreamDone(completion=completion)


class TestStreaming:
    async def test_stream_emite_deltas_y_texto_final(self, tmp_path):
        provider = FakeStreamingProvider([text_completion("hola mundo")])
        collector = EventCollector()
        agent = make_agent(tmp_path, provider, stream=True, on_event=collector)

        result = await agent.run("x")

        assert result == "hola mundo"
        deltas = collector.of_type(MessageDelta)
        assert "".join(delta.text for delta in deltas) == "hola mundo"
        assert len(deltas) > 1  # realmente llegó fragmentado

    async def test_stream_con_tool_calls_continua_el_ciclo(self, tmp_path):
        provider = FakeStreamingProvider(
            [tool_completion(SAY_CALL), text_completion("listo")]
        )
        agent = make_agent(tmp_path, provider, stream=True)

        result = await agent.run("x")

        assert result == "listo"
        assert provider.requests[1]["messages"][-1] == ChatMessage(
            role="tool", content="said: hola", tool_call_id="c1"
        )

    async def test_sin_flag_stream_usa_complete(self, tmp_path):
        provider = FakeStreamingProvider([text_completion("hi")])
        collector = EventCollector()
        agent = make_agent(tmp_path, provider, on_event=collector)  # stream=False

        await agent.run("x")

        assert collector.of_type(MessageDelta) == []

    async def test_fallback_si_el_provider_no_tiene_stream(self, tmp_path):
        provider = FakeProvider([text_completion("hi")])  # sin stream_complete
        agent = make_agent(tmp_path, provider, stream=True)

        assert await agent.run("x") == "hi"  # no explota: cae a complete()
