"""Pruebas de la API Web (FastAPI) levantando uvicorn real en un puerto efímero.

Por qué un servidor real y no `httpx.ASGITransport`: el transporte de ASGI
acumula todo el cuerpo de la respuesta antes de entregarlo, así que un stream
SSE "no streamería" — y los tests de confirmación necesitan enviar su
decisión MIENTRAS leen el stream (exactamente lo que hace el navegador).
Con uvicorn sobre 127.0.0.1 se prueba además la integración real que usa
`--web`.

Cubren el contrato de la Fase 4: creación de sesión (incluidos los errores
de configuración devueltos como 400 legibles), el stream SSE de un turno, el
flujo de confirmación aprobado/rechazado (D-03) y el bloqueo de turno
concurrente (409). Todo con `FakeProvider`: sin red ni LLM (RF-13).
"""

import asyncio
import json
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn

from ai_agent.config import Settings
from ai_agent.providers.base import ChatMessage, Completion, ToolCall
from ai_agent.tools.registry import Tool, ToolRegistry
from ai_agent.web.app import create_app

ENV_VARS_TO_CLEAR = ("AGENT_PROVIDER", "AGENT_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for variable in ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(variable, raising=False)


class FakeProvider:
    """Proveedor guionizado; `gate` permite bloquear el turno para probar 409."""

    model = "fake"

    def __init__(self, completions, gate: asyncio.Event | None = None) -> None:
        self._completions = list(completions)
        self.gate = gate
        self.closed = False

    async def complete(self, messages, tools=None) -> Completion:
        if self.gate is not None:
            await self.gate.wait()
        assert self._completions, "FakeProvider se quedó sin respuestas"
        return self._completions.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def text_completion(text: str) -> Completion:
    return Completion(message=ChatMessage(role="assistant", content=text))


def tool_completion(*calls: ToolCall) -> Completion:
    return Completion(
        message=ChatMessage(role="assistant", content=None, tool_calls=list(calls))
    )


APPEND_CALL = ToolCall(
    id="call-1", name="append_file", arguments='{"file_path": "f.txt", "content": "X"}'
)


def free_port() -> int:
    """Reserva un puerto efímero libre devolviéndolo inmediatamente."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def make_client(app, *, timeout: float = 60.0):
    """Levanta `app` con uvicorn y entrega un cliente HTTP apuntando a ella."""
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("uvicorn no llegó a arrancar")
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", timeout=timeout
        ) as client:
            yield client
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=10)


async def create_session(client, **overrides) -> str:
    response = await client.post("/api/sessions", json=overrides)
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


async def collect_events(response) -> list[dict]:
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def app(tmp_path):
    """App con FakeProvider sin gate y las herramientas por defecto reales."""
    settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)
    holder = {}

    def provider_factory(config, s):
        provider = FakeProvider(holder["completions"])
        holder["created"] = provider
        return provider

    application = create_app(settings, provider_factory=provider_factory)
    application.state.holder = holder
    return application


@pytest.fixture
def with_completions():
    """Asigna el guion de respuestas del FakeProvider de una app."""

    def setter(app, completions):
        app.state.holder["completions"] = completions

    return setter


class TestSesiones:
    async def test_creacion_exitosa(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            session_id = await create_session(client)
            listing = (await client.get("/api/sessions")).json()
            assert session_id in [s["session_id"] for s in listing]
            assert listing[0]["provider"] == "ollama"
            assert listing[0]["model"] == "qwen3:8b"

    async def test_workdir_invalido_devuelve_400(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            response = await client.post(
                "/api/sessions", json={"workdir": str(tmp_path / "no-existe")}
            )
            assert response.status_code == 400
            assert "no existe" in response.json()["detail"]

    async def test_proveedor_sin_clave_devuelve_400(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            response = await client.post("/api/sessions", json={"provider": "openrouter"})
            assert response.status_code == 400
            assert "OPENROUTER_API_KEY" in response.json()["detail"]

    async def test_borrar_sesion_cierra_proveedor(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            session_id = await create_session(client)
            response = await client.delete(f"/api/sessions/{session_id}")
            assert response.status_code == 200
            assert app.state.holder["created"].closed is True
            assert (await client.get("/api/sessions")).json() == []


class TestTurnos:
    async def test_turno_simple_stream_sse(self, tmp_path, app, with_completions):
        with_completions(app, [text_completion("hola desde fake")])
        async with make_client(app) as client:
            session_id = await create_session(client)
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json={"text": "decí hola"}
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("text/event-stream")
                events = await collect_events(response)
            types = [event["type"] for event in events]
            assert "FinalMessage" in types
            assert types[-1] == "turn_end"
            assert events[-1]["ok"] is True
            final = next(event for event in events if event["type"] == "FinalMessage")
            assert final["content"] == "hola desde fake"

    async def test_mensaje_vacio_400(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            session_id = await create_session(client)
            response = await client.post(
                f"/api/sessions/{session_id}/messages", json={"text": "   "}
            )
            assert response.status_code == 400

    async def test_sesion_desconocida_404(self, tmp_path, app):
        async with make_client(app) as client:
            response = await client.post(
                "/api/sessions/noexiste/messages", json={"text": "hola"}
            )
            assert response.status_code == 404

    async def test_error_del_proveedor_llega_como_turn_end(self, tmp_path, app, with_completions):
        class BrokenProvider(FakeProvider):
            async def complete(self, messages, tools=None):
                raise ConnectionError("sin conexión")

        settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)
        app_rota = create_app(
            settings, provider_factory=lambda config, s: BrokenProvider([])
        )
        async with make_client(app_rota) as client:
            session_id = await create_session(client)
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json={"text": "x"}
            ) as response:
                events = await collect_events(response)
            assert events[-1]["type"] == "turn_end"
            assert events[-1]["ok"] is False
            assert "ConnectionError" in events[-1]["error"]

    async def test_turno_concurrente_devuelve_409(self, tmp_path, with_completions):
        gate = asyncio.Event()
        settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)
        holder = {}

        def provider_factory(config, s):
            provider = FakeProvider(
                [text_completion("primero"), text_completion("segundo")], gate=gate
            )
            holder["created"] = provider
            return provider

        app = create_app(settings, provider_factory=provider_factory)

        async with make_client(app) as client:
            session_id = await create_session(client)

            async def primer_turno():
                async with client.stream(
                    "POST", f"/api/sessions/{session_id}/messages", json={"text": "a"}
                ) as response:
                    return await collect_events(response)

            pendiente = asyncio.create_task(primer_turno())
            await asyncio.sleep(0.3)  # el primer turno quedó bloqueado en el gate

            segunda = await client.post(
                f"/api/sessions/{session_id}/messages", json={"text": "b"}
            )
            assert segunda.status_code == 409

            gate.set()
            events = await asyncio.wait_for(pendiente, timeout=5)
            assert events[-1]["ok"] is True

            # La sesión quedó libre: el siguiente turno entra.
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json={"text": "c"}
            ) as response:
                events2 = await collect_events(response)
            assert events2[-1]["ok"] is True


class TestConfirmaciones:
    async def test_aprobada_ejecuta_la_herramienta(self, tmp_path):
        settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)

        def append_file(working_directory: str, file_path: str, content: str) -> str:
            (Path(working_directory) / file_path).write_text(content, encoding="utf-8")
            return f"appended to {file_path}"

        registry = ToolRegistry(
            [
                Tool(
                    name="append_file",
                    description="d",
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
                )
            ]
        )
        app = create_app(
            settings,
            provider_factory=lambda config, s: FakeProvider(
                [tool_completion(APPEND_CALL), text_completion("listo")]
            ),
            tool_factory=lambda: registry,
        )

        async with make_client(app) as client:
            session_id = await create_session(client)
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json={"text": "escribí"}
            ) as response:
                events = []
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[len("data: ") :])
                    events.append(event)
                    if event["type"] == "ConfirmationRequested":
                        decision = await client.post(
                            f"/api/sessions/{session_id}/confirmations",
                            json={"tool_call_id": event["tool_call_id"], "approved": True},
                        )
                        assert decision.status_code == 200
            assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "X"
            assert events[-1]["ok"] is True

    async def test_rechazada_no_ejecuta(self, tmp_path):
        settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)

        def append_file(working_directory: str, file_path: str, content: str) -> str:
            raise AssertionError("no debería ejecutarse tras un rechazo")

        registry = ToolRegistry(
            [
                Tool(
                    name="append_file",
                    description="d",
                    parameters={"type": "object", "properties": {}},
                    handler=append_file,
                    requires_confirmation=True,
                )
            ]
        )
        app = create_app(
            settings,
            provider_factory=lambda config, s: FakeProvider(
                [tool_completion(APPEND_CALL), text_completion("entendido")]
            ),
            tool_factory=lambda: registry,
        )

        async with make_client(app) as client:
            session_id = await create_session(client)
            async with client.stream(
                "POST", f"/api/sessions/{session_id}/messages", json={"text": "escribí"}
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event = json.loads(line[len("data: ") :])
                        if event["type"] == "ConfirmationRequested":
                            await client.post(
                                f"/api/sessions/{session_id}/confirmations",
                                json={
                                    "tool_call_id": event["tool_call_id"],
                                    "approved": False,
                                },
                            )
            assert not (tmp_path / "f.txt").exists()

    async def test_confirmacion_inexistente_404(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            session_id = await create_session(client)
            response = await client.post(
                f"/api/sessions/{session_id}/confirmations",
                json={"tool_call_id": "nada", "approved": True},
            )
            assert response.status_code == 404


class TestUI:
    async def test_index_se_ve(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            response = await client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
            assert "ai-agent" in response.text

    async def test_assets(self, tmp_path, app, with_completions):
        with_completions(app, [])
        async with make_client(app) as client:
            for asset in ("/static/app.js", "/static/styles.css"):
                response = await client.get(asset)
                assert response.status_code == 200, asset


class TestSerializacionEventos:
    """El contrato UI: nombre de dataclass en `type` + campos serializables."""

    def test_evento_a_dict(self):
        from ai_agent.events import ToolCallReceived
        from ai_agent.web.app import event_to_dict

        event = ToolCallReceived(
            tool_call_id="c1",
            tool_name="say",
            raw_arguments='{"text": "h"}',
            arguments={"text": "h"},
            requires_confirmation=False,
        )
        data = event_to_dict(event)
        assert data["type"] == "ToolCallReceived"
        assert data["tool_name"] == "say"
        assert data["arguments"] == {"text": "h"}
        assert json.dumps(data, ensure_ascii=False)  # serializable en SSE

    def test_formato_sse(self):
        from ai_agent.web.app import sse

        assert sse({"type": "x"}) == 'data: {"type": "x"}\n\n'
