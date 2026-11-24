"""Pruebas del REPL: comandos, cambios en caliente y sesiones.

El REPL se ejercita inyectando una fábrica de proveedores falsa, de modo que
los tests no tocan la red (RF-13). El bucle de lectura de `input()` no se
prueba aquí (es I/O pura de terminal); se verifica `handle_command` y el
armado del `Agent`, que contienen toda la lógica.
"""

import pytest

from ai_agent.cli import Repl
from ai_agent.config import Settings
from ai_agent.providers.base import ChatMessage, Completion
from ai_agent.tools.registry import ToolContext

ENV_VARS_TO_CLEAR = (
    "AGENT_PROVIDER",
    "AGENT_MODEL",
    "AGENT_BASE_URL",
    "AGENT_API_KEY",
    "AGENT_WORKDIR",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for variable in ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(variable, raising=False)


class FakeProvider:
    model = "fake"

    def __init__(self, model: str) -> None:
        self.model = model
        self.closed = False
        self.requests = []

    async def complete(self, messages, tools=None) -> Completion:
        self.requests.append(list(messages))
        return Completion(message=ChatMessage(role="assistant", content="ok"))

    async def aclose(self) -> None:
        self.closed = True


class ProviderRecorder:
    """Fábrica de proveedores falsos que recuerda los creados."""

    def __init__(self) -> None:
        self.created: list[FakeProvider] = []

    def __call__(self, provider_config, settings) -> FakeProvider:
        provider = FakeProvider(provider_config.model)
        self.created.append(provider)
        return provider


def make_repl(tmp_path, provider: str = "ollama", **settings_kwargs) -> tuple[Repl, ProviderRecorder]:
    settings = Settings(_env_file=None, provider=provider, workdir=tmp_path, **settings_kwargs)
    recorder = ProviderRecorder()
    repl = Repl(
        settings,
        sessions_dir=tmp_path / ".sessions",
        provider_factory=recorder,
    )
    return repl, recorder


class TestSetup:
    async def test_construye_agente_y_proveedor(self, tmp_path):
        repl, recorder = make_repl(tmp_path)

        await repl.setup()

        assert repl.agent is not None
        assert repl.provider_config.model == "qwen3:8b"  # default del preset ollama
        assert len(recorder.created) == 1

    async def test_turno_de_conversacion_funciona(self, tmp_path):
        repl, recorder = make_repl(tmp_path)
        await repl.setup()

        assert await repl.agent.run("hola") == "ok"
        assert len(repl.agent.history) == 3  # system + user + assistant


class TestComandos:
    async def test_ayuda_lista_comandos(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()

        assert await repl.handle_command("/ayuda") is False
        salida = capsys.readouterr().out
        for comando in ("/modelo", "/proveedor", "/directorio", "/guardar", "/cargar", "/salir"):
            assert comando in salida

    async def test_comando_desconocido(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()

        await repl.handle_command("/volar")
        assert "Comando desconocido" in capsys.readouterr().out

    async def test_salir_devuelve_true(self, tmp_path):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        assert await repl.handle_command("/salir") is True

    async def test_limpiar_vacia_historial(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        await repl.agent.run("hola")

        await repl.handle_command("/limpiar")

        assert len(repl.agent.history) == 1  # solo system
        assert repl.agent.history[0].role == "system"
        assert "vaciado" in capsys.readouterr().out

    async def test_directorio_cambia_el_sandbox(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        otro = tmp_path / "otro"
        otro.mkdir()

        await repl.handle_command(f"/directorio {otro}")

        assert repl.agent.context == ToolContext(
            working_directory=str(otro.resolve()),
            max_file_chars=repl.settings.max_file_chars,
        )
        assert "cambiado" in capsys.readouterr().out

    async def test_directorio_invalido_no_cambia_nada(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        anterior = repl.agent.context

        await repl.handle_command("/directorio /no/existe")

        assert repl.agent.context == anterior
        assert "no existe" in capsys.readouterr().out

    async def test_modelo_en_caliente_conserva_historial(self, tmp_path, capsys):
        repl, recorder = make_repl(tmp_path)
        await repl.setup()
        await repl.agent.run("hola")
        historial_antes = list(repl.agent.history)

        await repl.handle_command("/modelo qwen3:4b")

        assert repl.provider_config.model == "qwen3:4b"
        assert recorder.created[0].closed is True  # el anterior se cierra
        assert recorder.created[1] is repl.provider
        assert repl.agent.history == historial_antes
        assert "conserva" in capsys.readouterr().out

    async def test_proveedor_sin_clave_da_error_sin_romper(self, tmp_path, capsys):
        repl, recorder = make_repl(tmp_path)
        await repl.setup()
        modelo_antes = repl.provider_config.model

        await repl.handle_command("/proveedor openrouter")  # sin OPENROUTER_API_KEY

        assert "Error de configuración" in capsys.readouterr().out
        assert repl.provider_config.model == modelo_antes  # no cambió
        assert recorder.created[0].closed is False  # el proveedor activo sigue vivo

    async def test_proveedor_con_clave_cambia_y_resetea_modelo(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        repl, _ = make_repl(tmp_path)
        await repl.setup()

        await repl.handle_command("/proveedor openrouter")

        assert repl.provider_config.provider == "openrouter"
        assert repl.provider_config.model == "openrouter/free"  # default del preset


class TestSesiones:
    async def test_guardar_listar_y_cargar(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        await repl.agent.run("hola")
        historial_original = list(repl.agent.history)

        await repl.handle_command("/guardar demo")
        capsys.readouterr()
        await repl.handle_command("/sesiones")
        listado = capsys.readouterr().out
        assert "demo" in listado

        await repl.handle_command("/limpiar")
        capsys.readouterr()
        await repl.handle_command("/cargar demo")

        assert repl.agent.history == historial_original
        assert "Sesión cargada" in capsys.readouterr().out

    async def test_cargar_inexistente(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()

        await repl.handle_command("/cargar nada")
        assert "No se encontró" in capsys.readouterr().out

    async def test_guardar_dos_veces_actualiza_la_misma_sesion(self, tmp_path, capsys):
        repl, _ = make_repl(tmp_path)
        await repl.setup()
        await repl.agent.run("uno")
        await repl.handle_command("/guardar")
        capsys.readouterr()

        await repl.agent.run("dos")
        await repl.handle_command("/guardar")
        capsys.readouterr()

        assert len(repl.sessions.list_sessions()) == 1
        assert repl.sessions.list_sessions()[0].message_count == 4  # 2 turnos user+assistant
