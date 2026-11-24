"""Interfaz Web del agente: FastAPI + SSE con frontend vanilla (Fase 4, RF-10).

Arquitectura (decisión D-10):

- Una sesión en memoria por chat (`WebSession`): cada una tiene su propio
  `Agent`, su proveedor y su cola de confirmaciones pendientes. Es una app
  local monousuario; la persistencia larga vive en `.sessions/` (D-11).
- Un POST por turno cuya RESPUESTA es el stream SSE: el navegador recibe los
  eventos del núcleo uno a uno (incluidos los `MessageDelta` del streaming).
  Evita ids de turno, conexiones largas y reconexiones: petición = turno.
- La confirmación es el único canal inverso: el SSE entrega
  `ConfirmationRequested`, la UI muestra botones, y la decisión vuelve por
  `POST /confirmations`, que resuelve el `Future` que el `confirm_handler`
  del agente tiene en pausa. Si el cliente se desconecta a mitad de turno, el
  `finally` del generador cancela la tarea y deniega lo pendiente.

Seguridad: escucha solo en 127.0.0.1 y no tiene autenticación; exponerla a
la red está explícitamente fuera de alcance. No se sirve nada del workdir:
solo los assets de `static/`.
"""

import asyncio
import dataclasses
import json
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_agent import __version__
from ai_agent.agent import Agent, ConfirmationRequest
from ai_agent.cli import default_provider_factory
from ai_agent.config import ConfigError, Settings, resolve_provider_config
from ai_agent.events import AgentEvent
from ai_agent.tools import build_default_registry

STATIC_DIR = Path(__file__).parent / "static"

# --- Modelos de request/response --------------------------------------------


class SessionCreate(BaseModel):
    """Configuración opcional por sesión; lo ausente viene de `.env`/defaults."""

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    workdir: str | None = None


class MessageIn(BaseModel):
    text: str


class ConfirmationIn(BaseModel):
    tool_call_id: str
    approved: bool


# --- Sesiones ----------------------------------------------------------------


class WebSession:
    """Estado de un chat: agente, proveedor y confirmaciones pendientes."""

    def __init__(
        self,
        *,
        agent: Agent,
        provider: Any,
        provider_config: Any,
        settings: Settings,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.agent = agent
        self.provider = provider
        self.provider_config = provider_config
        self.settings = settings
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.busy = False
        # tool_call_id → future(bool) que el confirm_handler espera.
        self.pending_confirmations: dict[str, asyncio.Future[bool]] = {}

    async def aclose(self) -> None:
        for future in self.pending_confirmations.values():
            if not future.done():
                future.cancel()
        self.pending_confirmations.clear()
        await self.provider.aclose()


class SessionManager:
    """Registro en memoria de sesiones activas."""

    def __init__(self) -> None:
        self.sessions: dict[str, WebSession] = {}

    def get(self, session_id: str) -> WebSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"No existe la sesión '{session_id}'. Conectá de nuevo.",
            )
        return session

    async def close_all(self) -> None:
        for session in list(self.sessions.values()):
            await session.aclose()
        self.sessions.clear()


# --- Serialización de eventos -------------------------------------------------


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Convierte un evento del núcleo al JSON que consume la UI.

    El `type` es el nombre de la dataclass (`ToolCallReceived`, `MessageDelta`,
    ...): es el contrato estable entre el núcleo y `static/app.js`.
    """
    data = {"type": type(event).__name__}
    data.update(dataclasses.asdict(event))
    return data


def sse(data: dict[str, Any]) -> str:
    """Da formato de un mensaje Server-Sent Event."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- Aplicación ----------------------------------------------------------------


def create_app(
    base_settings: Settings | None = None,
    *,
    provider_factory: Callable = default_provider_factory,
    tool_factory: Callable = build_default_registry,
) -> FastAPI:
    """Factoría de la app FastAPI.

    Las fábricas son inyectables para poder probar toda la API con un
    `FakeProvider` y un registro de juguete, sin red ni claves (RF-13).
    """
    defaults = base_settings if base_settings is not None else Settings()
    manager = SessionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.close_all()

    app = FastAPI(
        title="ai-agent Web",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # --- infraestructura -------------------------------------------------------

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True, "version": __version__, "sessions": len(manager.sessions)}

    # --- sesiones ----------------------------------------------------------------

    @app.post("/api/sessions")
    async def create_session(body: SessionCreate) -> dict:
        overrides: dict[str, Any] = {}
        if body.provider is not None:
            overrides["provider"] = body.provider
        if body.model is not None:
            overrides["model"] = body.model
        if body.base_url is not None:
            overrides["base_url"] = body.base_url
        if body.workdir is not None:
            overrides["workdir"] = Path(body.workdir)
        settings = defaults.model_copy(update=overrides)

        try:
            provider_config = resolve_provider_config(settings)
        except ConfigError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        workdir = Path(settings.workdir)
        if not workdir.is_dir():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"El directorio de trabajo '{workdir}' no existe o no es un directorio."
                ),
            )

        provider = provider_factory(provider_config, settings)
        agent = Agent(
            provider=provider,
            tools=tool_factory(),
            workdir=workdir,
            max_iterations=settings.max_iterations,
            max_file_chars=settings.max_file_chars,
            stream=True,
        )
        session = WebSession(
            agent=agent,
            provider=provider,
            provider_config=provider_config,
            settings=settings,
        )
        manager.sessions[session.id] = session
        return {
            "session_id": session.id,
            "provider": provider_config.provider,
            "model": provider_config.model,
            "workdir": str(workdir.resolve()),
            "tools": agent.tools.names(),
        }

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        return [
            {
                "session_id": session.id,
                "provider": session.provider_config.provider,
                "model": session.provider_config.model,
                "workdir": session.agent.context.working_directory,
                "busy": session.busy,
                "created_at": session.created_at,
            }
            for session in manager.sessions.values()
        ]

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict:
        session = manager.get(session_id)
        await session.aclose()
        manager.sessions.pop(session_id, None)
        return {"ok": True}

    # --- turnos de conversación ----------------------------------------------------

    @app.post("/api/sessions/{session_id}/messages")
    async def post_message(session_id: str, body: MessageIn):
        """Corre un turno y devuelve SUS eventos como stream SSE.

        El turno del agente vive dentro del generador de respuesta: si el
        cliente corta la conexión, el generador se cancela y su `finally`
        cancela el turno, deniega confirmaciones pendientes y libera la
        sesión. Un turno a la vez por sesión (409 si está ocupada), que es lo
        que hace seguro reconfigurar los callbacks del `Agent` con
        `set_callbacks`.
        """
        session = manager.get(session_id)
        text = body.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="El mensaje está vacío.")
        if session.busy:
            raise HTTPException(
                status_code=409,
                detail="La sesión tiene un turno activo; esperá a que termine.",
            )
        session.busy = True

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

            async def on_event(event: AgentEvent) -> None:
                await queue.put(event_to_dict(event))

            async def confirm_handler(request: ConfirmationRequest) -> bool:
                future: asyncio.Future[bool] = (
                    asyncio.get_running_loop().create_future()
                )
                session.pending_confirmations[request.tool_call_id] = future
                try:
                    return await future
                finally:
                    session.pending_confirmations.pop(request.tool_call_id, None)

            session.agent.set_callbacks(
                on_event=on_event, confirm_handler=confirm_handler
            )
            turn = asyncio.create_task(session.agent.run(text))
            try:
                # Consume la cola mientras el turno viva o queden eventos.
                while not turn.done() or not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    yield sse(event)
                if turn.cancelled():
                    yield sse(
                        {
                            "type": "turn_end",
                            "ok": False,
                            "error": "El turno fue cancelado.",
                        }
                    )
                elif (error := turn.exception()) is not None:
                    yield sse(
                        {
                            "type": "turn_end",
                            "ok": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                else:
                    yield sse({"type": "turn_end", "ok": True})
            finally:
                if not turn.done():
                    turn.cancel()
                session.agent.set_callbacks()
                # Cliente desconectado o turno cerrado: nada queda en pausa.
                for future in session.pending_confirmations.values():
                    if not future.done():
                        future.set_result(False)
                session.pending_confirmations.clear()
                session.busy = False

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Evita buffering intermedio (nginx) para el streaming en vivo.
                "X-Accel-Buffering": "no",
            },
        )

    # --- confirmaciones -------------------------------------------------------------

    @app.post("/api/sessions/{session_id}/confirmations")
    async def resolve_confirmation(session_id: str, body: ConfirmationIn) -> dict:
        session = manager.get(session_id)
        future = session.pending_confirmations.get(body.tool_call_id)
        if future is None or future.done():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No hay una confirmación pendiente con ese id "
                    "(ya fue resuelta o el turno terminó)."
                ),
            )
        future.set_result(body.approved)
        return {"ok": True, "tool_call_id": body.tool_call_id, "approved": body.approved}

    # --- UI estática --------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
