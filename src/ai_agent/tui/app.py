"""Aplicación TUI del agente (textual): chat con actividad en vivo (RF-09).

Arquitectura: la TUI es una capa fina sobre el núcleo — no contiene lógica de
agente. Construye el `Agent` con `stream=True` y dos callbacks:

- `on_event` (asíncrono): traduce cada evento del núcleo a widgets del chat.
  Como los workers de textual corren en el mismo event loop que la app, el
  callback puede montar/actualizar widgets directamente.
- `confirm_handler`: abre el `ConfirmationDialog` modal con
  `push_screen_wait` desde el worker del turno y devuelve la decisión.

Cada turno del usuario corre en un worker (`@work(exclusive=True)`) para no
congelar la interfaz mientras el modelo genera o una herramienta bloquea
(RNF-05); el `Input` se deshabilita durante el turno.

Comandos soportados en esta fase: `/ayuda`, `/limpiar`, `/salir`. La gestión
de proveedor/modelo/sesiones vive en el REPL (`cli.py`); aquí se configura al
lanzar con los flags de la CLI (`--provider`, `--model`, `--workdir`, ...).
"""

import asyncio
import json
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Input, Footer, Static
from textual.containers import VerticalScroll

from ai_agent import __version__
from ai_agent.agent import Agent, ConfirmationRequest
from ai_agent.cli import default_provider_factory
from ai_agent.config import ConfigError, Settings, resolve_provider_config
from ai_agent.events import (
    AgentEvent,
    ConfirmationResolved,
    FinalMessage,
    LLMResponseReceived,
    MaxIterationsReached,
    MessageDelta,
    ToolCallReceived,
    ToolExecuted,
)
from ai_agent.providers.base import LLMProvider
from ai_agent.tui.dialogs import ConfirmationDialog
from ai_agent.tui.widgets import AssistantMessage, ToolActivity, UserMessage
from ai_agent.tools import build_default_registry

TUI_HELP = (
    "Comandos: /ayuda (esta ayuda) · /limpiar (vaciar historial y chat) · "
    "/salir (cerrar). Atajos: Ctrl+Q salir, Ctrl+L limpiar."
)

PREVIEW_MAX_CHARS = 400
PREVIEW_MAX_LINES = 6


def preview(text: str) -> str:
    """Recorta un resultado de herramienta para que no inunde el chat."""
    lines = text.splitlines() or [""]
    if len(lines) > PREVIEW_MAX_LINES:
        lines = [*lines[:PREVIEW_MAX_LINES], f"… ({len(lines) - PREVIEW_MAX_LINES} líneas más)"]
    result = "\n".join(lines)
    if len(result) > PREVIEW_MAX_CHARS:
        result = result[:PREVIEW_MAX_CHARS] + "…"
    return result


class ChatApp(App):
    """Chat TUI del agente: conversación, herramientas en vivo y confirmaciones."""

    TITLE = f"ai-agent {__version__}"

    CSS = """
    #chat-view {
        height: 1fr;
        border: round $accent;
        padding: 0 1;
        margin: 0 1;
    }
    Input {
        margin: 0 1;
    }
    .welcome {
        color: $text-muted;
        margin: 1 0;
    }
    .user {
        margin: 1 0 0 6;
        padding: 0 1;
        border-left: thick $primary;
        background: $surface;
    }
    .assistant {
        margin: 1 6 0 0;
        padding: 0 1;
        border-left: thick $success;
    }
    .tool {
        color: $text-muted;
        margin: 0 0 0 3;
    }
    .tool-error {
        color: $error;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Salir"),
        Binding("ctrl+l", "clear_history", "Limpiar"),
    ]

    def __init__(
        self,
        settings: Settings,
        *,
        auto_yes: bool = False,
        provider_factory=default_provider_factory,
        tool_factory=build_default_registry,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.auto_yes = auto_yes
        self._provider_factory = provider_factory
        self._tool_factory = tool_factory
        self.provider: LLMProvider | None = None
        self.provider_config = None
        self.agent: Agent | None = None
        # Burbuja de asistente en streaming; se renueva tras cada fase de
        # herramientas para que el flujo del chat quede en orden cronológico.
        self._current_response: AssistantMessage | None = None
        self._current_worker = None

    # --- ciclo de vida de la app -------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-view"):
            yield Static(
                "Agente listo. Escribí un mensaje para comenzar; /ayuda para comandos.",
                classes="welcome",
            )
        yield Input(placeholder="Mensaje para el agente… (Ctrl+Q para salir)")
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.provider_config = resolve_provider_config(self.settings)
            workdir = Path(self.settings.workdir)
            if not workdir.is_dir():
                raise ConfigError(
                    f"El directorio de trabajo '{workdir}' no existe o no es un directorio."
                )
            self.provider = self._provider_factory(self.provider_config, self.settings)
        except ConfigError as error:
            self.notify(f"Error de configuración: {error}", severity="error", timeout=None)
            self.exit()
            return
        self.agent = Agent(
            provider=self.provider,
            tools=self._tool_factory(),
            workdir=self.settings.workdir,
            max_iterations=self.settings.max_iterations,
            max_file_chars=self.settings.max_file_chars,
            stream=True,
            on_event=self.handle_agent_event,
            confirm_handler=self.confirm_tool,
        )
        self.sub_title = (
            f"{self.provider_config.provider} · {self.provider_config.model} · "
            f"{workdir.resolve()}"
        )
        self.query_one(Input).focus()

    # --- entrada del usuario --------------------------------------------------

    @on(Input.Submitted)
    async def handle_input(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if text.startswith("/"):
            await self.handle_command(text)
            return
        await self.query_one("#chat-view").mount(UserMessage(text))
        self._current_response = None
        input_widget = self.query_one(Input)
        input_widget.disabled = True
        self._current_worker = self.run_turn(text)

    @work(exclusive=True)
    async def run_turn(self, text: str) -> None:
        """Corre un turno del agente en background sin congelar la interfaz."""
        input_widget = self.query_one(Input)
        try:
            await self.agent.run(text)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # errores de red/API no deben matar la app
            await self.query_one("#chat-view").mount(
                ToolActivity(
                    f"Error: {type(error).__name__}: {error}", extra_class="tool-error"
                )
            )
        finally:
            input_widget.disabled = False
            input_widget.focus()
            self.query_one("#chat-view").scroll_end(animate=False)

    async def handle_command(self, line: str) -> None:
        command = line.split(maxsplit=1)[0].lower()
        chat_view = self.query_one("#chat-view")
        match command:
            case "/ayuda" | "/help":
                await chat_view.mount(ToolActivity(TUI_HELP))
            case "/limpiar":
                await self.action_clear_history()
                return  # action_clear_history ya scrollea
            case "/salir" | "/exit" | "/quit":
                await self.action_quit_app()
                return
            case _:
                await chat_view.mount(
                    ToolActivity(f"Comando desconocido: '{command}'. Usá /ayuda.")
                )
        chat_view.scroll_end(animate=False)

    # --- callbacks del núcleo --------------------------------------------------

    async def handle_agent_event(self, event: AgentEvent) -> None:
        """Traduce los eventos del núcleo a widgets del chat (RF-11)."""
        chat_view = self.query_one("#chat-view")
        match event:
            case MessageDelta(text=delta_text):
                if self._current_response is None:
                    self._current_response = AssistantMessage()
                    await chat_view.mount(self._current_response)
                self._current_response.append_delta(delta_text)
            case FinalMessage(content=content):
                if self._current_response is None:
                    self._current_response = AssistantMessage()
                    await chat_view.mount(self._current_response)
                self._current_response.set_final(content)
                self._current_response = None
            case ToolCallReceived():
                # Fin de la fase de texto actual: lo que el modelo diga
                # después irá en una burbuja nueva, debajo de las herramientas.
                self._current_response = None
                arguments = (
                    json.dumps(event.arguments, ensure_ascii=False)
                    if event.arguments is not None
                    else event.raw_arguments
                )
                warning = " ⚠ esperando confirmación" if event.requires_confirmation else ""
                await chat_view.mount(
                    ToolActivity(f"🔧 {event.tool_name} · {preview(arguments)}{warning}")
                )
            case ConfirmationResolved(approved=approved, tool_name=tool_name):
                verdict = f"✓ Aprobada: {tool_name}" if approved else f"✗ Rechazada: {tool_name}"
                await chat_view.mount(ToolActivity(verdict))
            case ToolExecuted():
                if not event.denied:
                    icon = "✗" if event.is_error else "↳"
                    extra = "tool-error" if event.is_error else ""
                    await chat_view.mount(
                        ToolActivity(f"{icon} {preview(event.result)}", extra_class=extra)
                    )
            case LLMResponseReceived():
                if self.settings.verbose and event.prompt_tokens is not None:
                    await chat_view.mount(
                        ToolActivity(
                            f"· iteración {event.iteration}: "
                            f"{event.prompt_tokens}+{event.completion_tokens} tokens"
                        )
                    )
            case MaxIterationsReached():
                await chat_view.mount(
                    ToolActivity(
                        f"⚠ Se alcanzó el máximo de {event.max_iterations} iteraciones.",
                        extra_class="tool-error",
                    )
                )
        chat_view.scroll_end(animate=False)

    async def confirm_tool(self, request: ConfirmationRequest) -> bool:
        """Abre el diálogo modal y espera la decisión del usuario.

        Se ejecuta dentro del worker del turno, que es el contexto exigido por
        `push_screen_wait`: el worker queda suspendido mientras la app sigue
        respondiendo al teclado y al mouse.
        """
        if self.auto_yes:
            return True
        return bool(await self.push_screen_wait(ConfirmationDialog(request)))

    # --- acciones -------------------------------------------------------------

    async def action_clear_history(self) -> None:
        self.agent.reset()
        self._current_response = None
        chat_view = self.query_one("#chat-view")
        chat_view.remove_children()
        await chat_view.mount(Static("Historial vaciado.", classes="welcome"))
        chat_view.scroll_end(animate=False)

    async def action_quit_app(self) -> None:
        if self.provider is not None:
            await self.provider.aclose()
        self.exit()
