"""Pruebas de la persistencia de sesiones (serialización, guardado, carga)."""

import json

import pytest

from ai_agent.providers.base import ChatMessage, ToolCall
from ai_agent.sessions import (
    SessionNotFoundError,
    SessionStore,
    message_from_dict,
    message_to_dict,
    new_session_id,
)

# Historial representativo: incluye un ciclo de herramienta completo
# (assistant con tool_calls + respuesta role="tool"), que es la parte más
# delicada de serializar: si se pierde, las APIs rechazan la conversación.
HISTORY = [
    ChatMessage(role="system", content="prompt del sistema"),
    ChatMessage(role="user", content="hola"),
    ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="c1", name="say", arguments='{"text": "hi"}')],
    ),
    ChatMessage(role="tool", content="said: hi", tool_call_id="c1"),
    ChatMessage(role="assistant", content="listo"),
]


def make_store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / ".sessions")


def write_raw_session(directory, session_id: str, **meta_overrides) -> None:
    """Escribe una sesión cruda controlando los metadatos (para orden y corrupción)."""
    meta = {
        "id": session_id,
        "name": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "provider": "ollama",
        "model": "qwen3:8b",
        "workdir": "/x",
        "message_count": 0,
        **meta_overrides,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{session_id}.json").write_text(
        json.dumps({"meta": meta, "messages": []}), encoding="utf-8"
    )


class TestSerializacion:
    def test_roundtrip_de_mensajes(self):
        for message in HISTORY:
            assert message_from_dict(message_to_dict(message)) == message

    def test_roundtrip_preserva_tool_calls(self):
        assistant = HISTORY[2]
        restored = message_from_dict(message_to_dict(assistant))
        assert restored.tool_calls[0].arguments == '{"text": "hi"}'

    def test_ids_unicos(self):
        assert new_session_id() != new_session_id()


class TestGuardarYCargar:
    def test_excluye_el_mensaje_del_sistema(self, tmp_path):
        store = make_store(tmp_path)

        meta = store.save(
            history=HISTORY, provider="ollama", model="qwen3:8b", workdir="/x"
        )

        data = json.loads(
            (tmp_path / ".sessions" / f"{meta.id}.json").read_text(encoding="utf-8")
        )
        roles = [message["role"] for message in data["messages"]]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert meta.message_count == 4

    def test_roundtrip_completo(self, tmp_path):
        store = make_store(tmp_path)

        meta = store.save(
            history=HISTORY, provider="p", model="m", workdir="/x", name="demo"
        )
        record = store.load(meta.id)

        assert record.messages == HISTORY[1:]  # sin el system
        assert record.meta.name == "demo"
        assert record.meta.provider == "p"

    def test_cargar_por_nombre(self, tmp_path):
        store = make_store(tmp_path)
        store.save(history=HISTORY, provider="p", model="m", workdir="/x", name="demo")

        record = store.load("demo")

        assert record.meta.name == "demo"

    def test_cargar_inexistente_da_error_accionable(self, tmp_path):
        with pytest.raises(SessionNotFoundError) as info:
            make_store(tmp_path).load("nada")
        assert "/sesiones" in str(info.value)

    def test_actualizar_conserva_id_y_created_at(self, tmp_path):
        store = make_store(tmp_path)
        original = store.save(history=HISTORY, provider="p", model="m", workdir="/x")

        updated = store.save(
            history=[*HISTORY, ChatMessage(role="user", content="más")],
            provider="p",
            model="m",
            workdir="/x",
            session_id=original.id,
            created_at=original.created_at,
        )

        assert updated.id == original.id
        assert updated.created_at == original.created_at
        assert updated.message_count == 5
        assert len(store.list_sessions()) == 1  # no duplica archivos


class TestListado:
    def test_ordena_por_actualizacion_descendente(self, tmp_path):
        directory = tmp_path / ".sessions"
        write_raw_session(directory, "vieja", updated_at="2026-01-01T00:00:00")
        write_raw_session(directory, "nueva", updated_at="2026-06-01T00:00:00")

        metas = SessionStore(directory).list_sessions()

        assert [meta.id for meta in metas] == ["nueva", "vieja"]

    def test_ignora_archivos_corruptos(self, tmp_path):
        directory = tmp_path / ".sessions"
        write_raw_session(directory, "valida")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rota.json").write_text("{no soy json", encoding="utf-8")

        metas = SessionStore(directory).list_sessions()

        assert [meta.id for meta in metas] == ["valida"]

    def test_directorio_inexistente_lista_vacio(self, tmp_path):
        assert SessionStore(tmp_path / "nada").list_sessions() == []

    def test_cargar_sesion_corrupta_da_error(self, tmp_path):
        directory = tmp_path / ".sessions"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rota.json").write_text("{no soy json", encoding="utf-8")

        with pytest.raises(SessionNotFoundError) as info:
            SessionStore(directory).load("rota")
        assert "corrupta" in str(info.value)
