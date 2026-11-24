"""Pruebas de la TUI con el pilot de textual (`App.run_test`).

Qué verifican: arranque con la configuración resuelta, render de un turno
simple (burbuja del usuario + respuesta del asistente), el diálogo modal de
confirmación con aprobación y rechazo (D-03) y el comando `/limpiar`. Todo
con un proveedor falso: sin red ni modelos reales (RF-13).
"""

from pathlib import Path

import pytest
from textual.widgets import Input

from ai_agent.config import Settings
from ai_agent.providers.base import ChatMessage, Completion, ToolCall, Usage
from ai_agent.tui.app import ChatApp
from ai_agent.tui.dialogs import ConfirmationDialog
from ai_agent.tui.widgets import AssistantMessage, UserMessage
from ai_agent.tools.registry import Tool, ToolRegistry

ENV_VARS_TO_CLEAR = ("AGENT_PROVIDER", "AGENT_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for variable in ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(variable, raising=False)


class FakeProvider:
    """Proveedor guionizado; sin `stream_complete`, el agente cae a complete()."""

    model = "fake"

    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)

    async def complete(self, messages, tools=None) -> Completion:
        assert self._completions, "FakeProvider se quedó sin respuestas"
        return self._completions.pop(0)

    async def aclose(self) -> None:
        pass


def text_completion(text: str) -> Completion:
    return Completion(message=ChatMessage(role="assistant", content=text))


def tool_completion(*calls: ToolCall) -> Completion:
    return Completion(
        message=ChatMessage(role="assistant", content=None, tool_calls=list(calls))
    )


def make_confirm_registry() -> ToolRegistry:
    def append_file(working_directory: str, file_path: str, content: str) -> str:
        path = Path(working_directory) / file_path
        previous = path.read_text(encoding="utf-8") if path.exists() else ""
        path.write_text(previous + content, encoding="utf-8")
        return f"appended to {file_path}"

    return ToolRegistry(
        [
            Tool(
                name="append_file",
                description="Agrega texto a un archivo",
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


APPEND_CALL = ToolCall(
    id="c1", name="append_file", arguments='{"file_path": "f.txt", "content": "X"}'
)


def make_app(tmp_path, completions, *, tools=None, auto_yes=False) -> ChatApp:
    provider = FakeProvider(completions)
    settings = Settings(_env_file=None, provider="ollama", workdir=tmp_path)
    kwargs = {}
    if tools is not None:
        kwargs["tool_factory"] = lambda: tools
    return ChatApp(
        settings,
        auto_yes=auto_yes,
        provider_factory=lambda config, s: provider,
        **kwargs,
    )


async def submit(pilot, app: ChatApp, text: str) -> None:
    """Escribe en el Input, lo envía y espera a que termine el worker del turno."""
    app.query_one(Input).value = text
    await pilot.press("enter")
    await pilot.pause()
    if app._current_worker is not None:
        await app._current_worker.wait()


async def wait_for_dialog(pilot, app: ChatApp) -> None:
    """Espera (con tope) a que el diálogo modal de confirmación aparezca."""
    for _ in range(100):
        await pilot.pause(0.01)
        if isinstance(app.screen, ConfirmationDialog):
            return
    raise AssertionError("el diálogo de confirmación no apareció")


class TestArranque:
    async def test_muestra_proveedor_y_modelo(self, tmp_path):
        app = make_app(tmp_path, [])

        async with app.run_test() as pilot:
            await pilot.pause()
            assert "ollama" in app.sub_title
            assert "qwen3:8b" in app.sub_title


class TestTurnoSimple:
    async def test_renderiza_usuario_y_respuesta(self, tmp_path):
        app = make_app(tmp_path, [text_completion("hola desde fake")])

        async with app.run_test() as pilot:
            await submit(pilot, app, "saludame")

            chat = app.query_one("#chat-view")
            assert any(
                isinstance(widget, UserMessage) and widget.user_text == "saludame"
                for widget in chat.children
            )
            responses = [w for w in chat.children if isinstance(w, AssistantMessage)]
            assert len(responses) == 1
            assert "hola desde fake" in responses[0]._buffer
            # El input queda habilitado para el siguiente turno
            assert app.query_one(Input).disabled is False


class TestConfirmacion:
    async def test_aprobacion_ejecuta_la_herramienta(self, tmp_path):
        app = make_app(
            tmp_path,
            [tool_completion(APPEND_CALL), text_completion("listo")],
            tools=make_confirm_registry(),
        )

        async with app.run_test() as pilot:
            app.query_one(Input).value = "escribe el archivo"
            await pilot.press("enter")
            await wait_for_dialog(pilot, app)
            await pilot.press("s")  # aprueba con la tecla s
            await app._current_worker.wait()

            assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "X"

    async def test_rechazo_no_toca_el_sistema(self, tmp_path):
        app = make_app(
            tmp_path,
            [tool_completion(APPEND_CALL), text_completion("entendido")],
            tools=make_confirm_registry(),
        )

        async with app.run_test() as pilot:
            app.query_one(Input).value = "escribe el archivo"
            await pilot.press("enter")
            await wait_for_dialog(pilot, app)
            await pilot.press("n")  # rechaza
            await app._current_worker.wait()

            assert not (tmp_path / "f.txt").exists()

    async def test_auto_yes_omite_el_dialogo(self, tmp_path):
        app = make_app(
            tmp_path,
            [tool_completion(APPEND_CALL), text_completion("listo")],
            tools=make_confirm_registry(),
            auto_yes=True,
        )

        async with app.run_test() as pilot:
            await submit(pilot, app, "escribe el archivo")

            assert (tmp_path / "f.txt").exists()


class TestComandos:
    async def test_limpiar_vacia_el_chat_y_el_historial(self, tmp_path):
        app = make_app(tmp_path, [text_completion("uno")])

        async with app.run_test() as pilot:
            await submit(pilot, app, "hola")
            assert len(app.agent.history) == 3  # system + user + assistant

            app.query_one(Input).value = "/limpiar"
            await pilot.press("enter")
            await pilot.pause()

            assert len(app.agent.history) == 1  # solo system
            chat = app.query_one("#chat-view")
            assert not any(
                isinstance(widget, UserMessage) for widget in chat.children
            )
