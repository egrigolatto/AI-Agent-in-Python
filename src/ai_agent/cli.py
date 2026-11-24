"""Entrada de línea de comandos del agente: modo "un tiro" y REPL multi-turno.

Dos modos de uso (RF-07):

- Con prompt posicional: `ai-agent "instrucción"` procesa un turno y termina.
- Sin prompt: `ai-agent` abre el REPL interactivo, con conversación
  multi-turno, comandos de gestión (`/modelo`, `/proveedor`, `/directorio`,
  ...) y sesiones guardables (`/guardar`, `/sesiones`, `/cargar`).

Cómo se conecta con el núcleo: la CLI construye `Settings` → resuelve
`ProviderConfig` → instancia `OpenAICompatibleProvider` y `ToolRegistry` →
crea el `Agent` inyectando dos callbacks: un impresor de eventos (`on_event`)
y un confirmador por terminal (`confirm_handler`). Toda la lógica de agente
vive en el núcleo; aquí solo hay I/O (RF-11). Cambiar de proveedor/modelo en
caliente reconstruye únicamente el proveedor: el `Agent` conserva su
historial, que es lo que hace posible la conversación multi-turno.
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from ai_agent import __version__
from ai_agent.agent import Agent, ConfirmationRequest
from ai_agent.config import (
    ConfigError,
    PROVIDER_PRESETS,
    Settings,
    resolve_provider_config,
)
from ai_agent.events import (
    AgentEvent,
    ConfirmationResolved,
    LLMResponseReceived,
    MaxIterationsReached,
    ToolCallReceived,
    ToolExecuted,
)
from ai_agent.providers.base import LLMProvider
from ai_agent.providers.openai_compat import OpenAICompatibleProvider
from ai_agent.sessions import (
    DEFAULT_SESSIONS_DIR,
    SessionNotFoundError,
    SessionStore,
)
from ai_agent.tools import build_default_registry
from ai_agent.tools.registry import ToolContext

YES_ANSWERS = {"s", "si", "sí", "y"}
EXIT_COMMANDS = {"/salir", "/exit", "/quit"}

HELP_TEXT = """\
Comandos disponibles:
  /ayuda                muestra esta ayuda
  /proveedor <nombre>   cambia el perfil de proveedor (openrouter, ollama, openai, custom)
  /modelo <modelo>      cambia el modelo del proveedor actual
  /directorio <ruta>    cambia el directorio de trabajo (sandbox)
  /limpiar              vacía el historial de la conversación
  /guardar [nombre]     guarda la sesión actual en .sessions/
  /sesiones             lista las sesiones guardadas
  /cargar <id|nombre>   retoma una sesión guardada
  /salir                termina el REPL (también con Ctrl-D)

Cualquier otra entrada se envía al agente como un turno de la conversación.\
"""


def build_arg_parser() -> argparse.ArgumentParser:
    """Define los argumentos de la CLI (tienen máxima prioridad, D-07)."""
    parser = argparse.ArgumentParser(
        prog="ai-agent",
        description=(
            "Agente de IA con herramientas de archivos, configurable para "
            "OpenRouter, Ollama y cualquier API compatible con OpenAI."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help=(
            "Instrucción para el agente. Si se omite, inicia el REPL "
            "interactivo multi-turno."
        ),
    )
    parser.add_argument(
        "--provider",
        help="Perfil de proveedor: openrouter (defecto), ollama, openai, custom",
    )
    parser.add_argument("--model", help="Modelo a usar (sobrescribe el del preset)")
    parser.add_argument("--base-url", help="Base URL de la API (sobrescribe el preset)")
    parser.add_argument(
        "--workdir",
        type=Path,
        help="Directorio de trabajo/sandbox del agente (defecto: ./calculator)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Muestra proveedor, tokens por iteración, argumentos y resultados de herramientas",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Lanza la interfaz gráfica de terminal (textual) en lugar de la CLI",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Lanza la interfaz Web (FastAPI) en http://127.0.0.1, sin autenticación",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Puerto para --web (defecto: 8000)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Aprueba automáticamente las herramientas peligrosas (usar con precaución)",
    )
    parser.add_argument(
        "--version", action="version", version=f"ai-agent {__version__}"
    )
    return parser


def build_settings(args: argparse.Namespace) -> Settings:
    """Combina `.env`/entorno con las sobrescrituras explícitas de la CLI."""
    # load_dotenv publica el .env en os.environ para que la resolución en
    # cascada también vea las variables legacy (OPENROUTER_API_KEY, D-07).
    load_dotenv()
    settings = Settings()
    overrides = {
        key: value
        for key, value in (
            ("provider", args.provider),
            ("model", args.model),
            ("base_url", args.base_url),
            ("workdir", args.workdir),
            ("verbose", True if args.verbose else None),
        )
        if value is not None
    }
    return settings.model_copy(update=overrides)


def make_event_printer(verbose: bool):
    """Crea el callback `on_event` que vuelca la actividad del agente a stdout."""

    def print_event(event: AgentEvent) -> None:
        match event:
            case ToolCallReceived():
                arguments = f" {event.arguments}" if verbose else ""
                print(f" - Herramienta solicitada: {event.tool_name}{arguments}")
                if event.is_error and event.error_detail:
                    print(f"   ! {event.error_detail}")
            case ConfirmationResolved():
                verdict = "Aprobada" if event.approved else "Rechazada"
                print(f" - {verdict}: {event.tool_name}")
            case ToolExecuted():
                if verbose:
                    indented = event.result.replace("\n", "\n   ")
                    print(f"   -> {indented}")
            case LLMResponseReceived():
                if verbose and event.prompt_tokens is not None:
                    print(
                        f" - Modelo {event.model} (iteración {event.iteration}): "
                        f"prompt={event.prompt_tokens} tokens, "
                        f"respuesta={event.completion_tokens} tokens"
                    )
            case MaxIterationsReached():
                print(
                    f"Error: el agente alcanzó {event.max_iterations} iteraciones "
                    "sin una respuesta final.",
                    file=sys.stderr,
                )

    return print_event


def make_confirm_handler(auto_yes: bool):
    """Crea el `confirm_handler` de la CLI: pregunta s/N por terminal (D-03)."""

    def confirm(request: ConfirmationRequest) -> bool:
        if auto_yes:
            return True
        summary = json.dumps(request.arguments, ensure_ascii=False)
        if len(summary) > 400:
            summary = summary[:400] + "…"
        print(
            f"\n[!] El agente quiere ejecutar '{request.tool_name}' "
            f"con argumentos: {summary}"
        )
        while True:
            answer = input("¿Aprobar ejecución? [s/N] ").strip().lower()
            if answer in YES_ANSWERS:
                return True
            if answer in {"", "n", "no"}:
                return False
            print("Responde 's' (sí) o 'n' (no).")

    return confirm


def default_provider_factory(provider_config, settings: Settings) -> LLMProvider:
    """Fábrica de proveedores por defecto; los tests inyectan una falsa."""
    return OpenAICompatibleProvider(
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
        model=provider_config.model,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
    )


class Repl:
    """Bucle interactivo multi-turno sobre el mismo núcleo `Agent`.

    Responsabilidades: leer líneas del usuario, despachar comandos `/...` y
    delegar todo lo demás al agente. Cambiar de proveedor o modelo en
    caliente reconstruye solo el provider (el historial vive en el `Agent` y
    sobrevive al cambio); cambiar de directorio actualiza el `ToolContext`.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        auto_yes: bool = False,
        provider_factory: Callable = default_provider_factory,
        sessions_dir: Path = DEFAULT_SESSIONS_DIR,
    ) -> None:
        self.settings = settings
        self.auto_yes = auto_yes
        self._provider_factory = provider_factory
        self.sessions = SessionStore(sessions_dir)
        self.provider: LLMProvider | None = None
        self.provider_config = None
        self.agent: Agent | None = None
        self.session_id: str | None = None
        self.session_created_at: str | None = None

    # --- ciclo de vida ------------------------------------------------------

    async def setup(self) -> None:
        """Construye proveedor y agente iniciales. Lanza ConfigError si algo falta."""
        self.provider_config = resolve_provider_config(self.settings)
        self._validate_workdir(self.settings.workdir)
        self.provider = self._provider_factory(self.provider_config, self.settings)
        self.agent = self._build_agent()

    def _validate_workdir(self, workdir: Path) -> None:
        if not Path(workdir).is_dir():
            raise ConfigError(
                f"El directorio de trabajo '{workdir}' no existe o no es un "
                "directorio. Cámbialo con --workdir, AGENT_WORKDIR o /directorio."
            )

    def _build_agent(self) -> Agent:
        return Agent(
            provider=self.provider,
            tools=build_default_registry(),
            workdir=self.settings.workdir,
            max_iterations=self.settings.max_iterations,
            max_file_chars=self.settings.max_file_chars,
            on_event=make_event_printer(self.settings.verbose),
            confirm_handler=make_confirm_handler(self.auto_yes),
        )

    async def start(self) -> int:
        """Bucle principal: lee entradas, despacha comandos y corre turnos."""
        await self.setup()
        print(f"ai-agent {__version__} — REPL multi-turno")
        print(
            f"Proveedor: {self.provider_config.provider} | "
            f"Modelo: {self.provider_config.model} | "
            f"Workdir: {Path(self.settings.workdir).resolve()}"
        )
        print("/ayuda para comandos; /salir o Ctrl-D para terminar.\n")

        while True:
            try:
                line = (await asyncio.to_thread(input, "tú> ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                if await self.handle_command(line):
                    break
                continue
            try:
                result = await self.agent.run(line)
            except Exception as error:  # errores de red/API no deben matar el REPL
                print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
                continue
            print(f"\n{result}\n")

        await self.provider.aclose()
        print("¡Hasta luego!")
        return 0

    # --- comandos -------------------------------------------------------------

    async def handle_command(self, line: str) -> bool:
        """Ejecuta un comando `/...`. Devuelve True si hay que terminar el REPL."""
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in EXIT_COMMANDS:
            return True
        match command:
            case "/ayuda" | "/help":
                print(HELP_TEXT)
            case "/limpiar":
                self.agent.reset()
                self.session_id = None
                self.session_created_at = None
                print("Historial vaciado. La conversación empieza de cero.")
            case "/proveedor":
                await self._switch_provider(argument)
            case "/modelo":
                await self._switch_model(argument)
            case "/directorio":
                self._switch_workdir(argument)
            case "/guardar":
                self._save_session(argument or None)
            case "/sesiones":
                self._list_sessions()
            case "/cargar":
                self._load_session(argument)
            case _:
                print(f"Comando desconocido: '{command}'. Usá /ayuda para ver los comandos.")
        return False

    async def _switch_provider(self, argument: str) -> None:
        if not argument:
            disponibles = ", ".join(sorted(PROVIDER_PRESETS))
            print(f"Uso: /proveedor <nombre>. Disponibles: {disponibles}.")
            return
        # Al cambiar de perfil, el modelo del proveedor anterior probablemente
        # no exista en el nuevo: se resetea al default del preset elegido.
        await self._rebuild_provider(
            self.settings.model_copy(update={"provider": argument, "model": None})
        )

    async def _switch_model(self, argument: str) -> None:
        if not argument:
            print(f"Modelo actual: {self.provider_config.model}. Uso: /modelo <nombre>.")
            return
        await self._rebuild_provider(self.settings.model_copy(update={"model": argument}))

    async def _rebuild_provider(self, new_settings: Settings) -> None:
        """Valida y aplica nueva configuración de proveedor sin perder el historial."""
        try:
            new_config = resolve_provider_config(new_settings)
        except ConfigError as error:
            print(f"Error de configuración: {error}")
            return
        old_provider = self.provider
        try:
            self.provider = self._provider_factory(new_config, new_settings)
        except Exception as error:
            print(f"No se pudo crear el proveedor: {error}")
            return
        self.provider_config = new_config
        self.settings = new_settings
        self.agent.provider = self.provider
        if old_provider is not None:
            await old_provider.aclose()
        print(
            f"Proveedor activo: {new_config.provider} | Modelo: {new_config.model} "
            "(el historial de la conversación se conserva)."
        )

    def _switch_workdir(self, argument: str) -> None:
        if not argument:
            print(
                f"Directorio actual: {Path(self.settings.workdir).resolve()}. "
                "Uso: /directorio <ruta>."
            )
            return
        new_workdir = Path(argument)
        if not new_workdir.is_dir():
            print(f"Error: '{argument}' no existe o no es un directorio.")
            return
        self.settings = self.settings.model_copy(update={"workdir": new_workdir})
        # Nuevo ToolContext con ruta absoluta: el sandbox cambia desde la
        # próxima herramienta; el historial se conserva.
        self.agent.context = ToolContext(
            working_directory=str(new_workdir.resolve()),
            max_file_chars=self.settings.max_file_chars,
        )
        print(f"Directorio de trabajo cambiado a: {new_workdir.resolve()}")

    # --- sesiones ---------------------------------------------------------------

    def _save_session(self, name: str | None) -> None:
        meta = self.sessions.save(
            history=self.agent.history,
            provider=self.provider_config.provider,
            model=self.provider_config.model,
            workdir=self.agent.context.working_directory,
            name=name or None,
            session_id=self.session_id,
            created_at=self.session_created_at,
        )
        self.session_id = meta.id
        self.session_created_at = meta.created_at
        label = f" (nombre: {meta.name})" if meta.name else ""
        print(
            f"Sesión guardada: {meta.id}{label} — {meta.message_count} mensajes "
            f"en {self.sessions.directory / (meta.id + '.json')}"
        )

    def _list_sessions(self) -> None:
        metas = self.sessions.list_sessions()
        if not metas:
            print(f"No hay sesiones guardadas en {self.sessions.directory.resolve()}.")
            return
        print("Sesiones guardadas (más recientes primero):")
        for meta in metas:
            label = f" «{meta.name}»" if meta.name else ""
            print(
                f"  {meta.id}{label}  {meta.provider}/{meta.model}  "
                f"{meta.message_count} msgs  actualizada {meta.updated_at}"
            )

    def _load_session(self, argument: str) -> None:
        if not argument:
            print("Uso: /cargar <id|nombre>. Usá /sesiones para ver las disponibles.")
            return
        try:
            record = self.sessions.load(argument)
        except SessionNotFoundError as error:
            print(f"Error: {error}")
            return
        system_message = self.agent.history[0]
        self.agent.history = [system_message, *record.messages]
        self.session_id = record.meta.id
        self.session_created_at = record.meta.created_at
        print(
            f"Sesión cargada: {record.meta.id} — {record.meta.message_count} mensajes."
        )
        warnings = []
        if record.meta.model != self.provider_config.model:
            warnings.append(
                f"se guardó con el modelo '{record.meta.model}' "
                f"(actual: '{self.provider_config.model}', cambialo con /modelo)"
            )
        if record.meta.provider != self.provider_config.provider:
            warnings.append(
                f"se guardó con el proveedor '{record.meta.provider}' "
                f"(actual: '{self.provider_config.provider}', cambialo con /proveedor)"
            )
        for warning in warnings:
            print(f"  Aviso: la sesión {warning}.")


async def run_agent(settings: Settings, prompt: str, auto_yes: bool) -> int:
    """Modo "un tiro": construye el agente, procesa un turno e imprime la respuesta."""
    repl = Repl(settings, auto_yes=auto_yes)
    await repl.setup()
    try:
        if settings.verbose:
            print(
                f"Proveedor: {repl.provider_config.provider} | "
                f"Modelo: {repl.provider_config.model} | "
                f"Workdir: {Path(settings.workdir).resolve()}"
            )
        result = await repl.agent.run(prompt)
        print(result)
        return 0
    finally:
        await repl.provider.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada (console script `ai-agent` y `python -m ai_agent`)."""
    args = build_arg_parser().parse_args(argv)
    if sum(bool(flag) for flag in (args.tui, args.web)) > 1:
        print("Error: --tui y --web son excluyentes.", file=sys.stderr)
        return 2
    if (args.tui or args.web) and args.prompt:
        print(
            f"Error: {'--tui' if args.tui else '--web'} no admite un prompt "
            "posicional; el mensaje se escribe dentro de la interfaz.",
            file=sys.stderr,
        )
        return 2
    try:
        settings = build_settings(args)
        if args.web:
            # Import tardío: las demás interfaces no pagan el costo de FastAPI.
            import uvicorn

            from ai_agent.web.app import create_app

            print(f"Interfaz Web: http://127.0.0.1:{args.port} (Ctrl+C para terminar)")
            uvicorn.run(
                create_app(settings),
                host="127.0.0.1",
                port=args.port,
                log_level="warning",
            )
            return 0
        if args.tui:
            # Import tardío: la CLI/REPL no pagan el costo de importar textual.
            from ai_agent.tui.app import ChatApp

            ChatApp(settings, auto_yes=args.yes).run()
            return 0
        if args.prompt is None:
            return asyncio.run(Repl(settings, auto_yes=args.yes).start())
        return asyncio.run(run_agent(settings, args.prompt, args.yes))
    except ConfigError as error:
        print(f"Error de configuración: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print()  # salta de línea tras ^C
        return 130
    except Exception as error:  # errores de red/API del proveedor
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        if args.verbose:
            raise
        return 1
