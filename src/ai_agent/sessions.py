"""Persistencia de sesiones de conversación en JSON (RF-08, decisión D-11).

Qué resuelve: un REPL sin persistencia pierde todo el contexto al cerrar el
proceso. Este módulo guarda el historial de una conversación (sin el mensaje
del sistema, que siempre se regenera con el prompt vigente) junto con los
metadatos necesarios para retomar el trabajo: proveedor, modelo y directorio
de trabajo.

Formato elegido: un archivo JSON por sesión en `.sessions/` (excluido de
Git). JSON en vez de SQLite porque es legible a simple vista, fácil de
depurar y suficiente para un usuario local; la migración a SQLite queda
documentada como posible si la interfaz Web la necesita.

Detalle importante de fidelidad: los `tool_calls` del asistente y los mensajes
`role="tool"` se serializan tal cual. Sin ellos, un historial con ciclos de
herramientas a medio terminar sería rechazado por las APIs (todo `tool_call`
exige su respuesta), así que la sesión cargada es idéntica a la original.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ai_agent.providers.base import ChatMessage, ToolCall

DEFAULT_SESSIONS_DIR = Path(".sessions")


class SessionNotFoundError(FileNotFoundError):
    """La sesión pedida no existe (ni por id ni por nombre)."""


def message_to_dict(message: ChatMessage) -> dict:
    """Serializa un mensaje del núcleo a un dict JSON-compatible."""
    data: dict = {"role": message.role}
    if message.content is not None:
        data["content"] = message.content
    if message.tool_calls:
        data["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in message.tool_calls
        ]
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    return data


def message_from_dict(data: dict) -> ChatMessage:
    """Reconstruye un `ChatMessage` desde su representación serializada."""
    tool_calls = [
        ToolCall(
            id=raw["id"], name=raw["name"], arguments=raw.get("arguments", "{}")
        )
        for raw in data.get("tool_calls", [])
    ]
    return ChatMessage(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tool_calls or None,
        tool_call_id=data.get("tool_call_id"),
    )


@dataclass(frozen=True)
class SessionMeta:
    """Metadatos de una sesión guardada (lo que se lista en `/sesiones`)."""

    id: str
    name: str | None
    created_at: str
    updated_at: str
    provider: str
    model: str
    workdir: str
    message_count: int


@dataclass(frozen=True)
class SessionRecord:
    """Sesión completa: metadatos + historial reconstruido."""

    meta: SessionMeta
    messages: list[ChatMessage]


def new_session_id() -> str:
    """Id legible y único: fecha-hora + sufijo aleatorio corto."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"


class SessionStore:
    """Almacén de sesiones: guardar, listar y cargar archivos JSON."""

    def __init__(self, directory: Path = DEFAULT_SESSIONS_DIR) -> None:
        self.directory = Path(directory)

    # --- escritura ------------------------------------------------------------

    def save(
        self,
        *,
        history: list[ChatMessage],
        provider: str,
        model: str,
        workdir: str | Path,
        name: str | None = None,
        session_id: str | None = None,
        created_at: str | None = None,
    ) -> SessionMeta:
        """Guarda (o actualiza) una sesión y devuelve sus metadatos.

        El mensaje del sistema se excluye: al cargar, el agente antepone el
        prompt vigente, de modo que actualizar el prompt no rompe las sesiones
        viejas ni conserva instrucciones obsoletas.
        """
        messages = [message for message in history if message.role != "system"]
        now = datetime.now().isoformat(timespec="seconds")
        session_id = session_id or new_session_id()
        meta = SessionMeta(
            id=session_id,
            name=name,
            created_at=created_at or now,
            updated_at=now,
            provider=provider,
            model=model,
            workdir=str(workdir),
            message_count=len(messages),
        )
        payload = {
            "meta": asdict(meta),
            "messages": [message_to_dict(message) for message in messages],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(session_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    # --- lectura ---------------------------------------------------------------

    def list_sessions(self) -> list[SessionMeta]:
        """Lista las sesiones válidas, más recientes primero.

        Los archivos corruptos se ignoran en lugar de romper el listado: una
        sesión dañada no debe inutilizar el resto.
        """
        metas: list[SessionMeta] = []
        if not self.directory.is_dir():
            return metas
        for path in self.directory.glob("*.json"):
            try:
                meta = SessionMeta(**json.loads(path.read_text(encoding="utf-8"))["meta"])
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
            metas.append(meta)
        metas.sort(key=lambda meta: meta.updated_at, reverse=True)
        return metas

    def load(self, id_or_name: str) -> SessionRecord:
        """Carga una sesión por id exacto o, si no, por nombre (la más reciente).

        Lanza `SessionNotFoundError` con un mensaje accionable si no existe.
        """
        path = self._path(id_or_name)
        if not path.is_file():
            matches = [
                meta for meta in self.list_sessions() if meta.name == id_or_name
            ]
            if not matches:
                raise SessionNotFoundError(
                    f"No se encontró ninguna sesión con id o nombre '{id_or_name}'. "
                    "Usá /sesiones para ver las disponibles."
                )
            path = self._path(matches[0].id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = SessionMeta(**data["meta"])
            messages = [message_from_dict(raw) for raw in data["messages"]]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise SessionNotFoundError(
                f"La sesión '{path.name}' está corrupta y no puede cargarse: {error}"
            ) from error
        return SessionRecord(meta=meta, messages=messages)

    # --- interno ---------------------------------------------------------------

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"
