"""Pruebas de la configuración: presets, cascada de claves y errores accionables."""

import pytest

from ai_agent.config import (
    ConfigError,
    Settings,
    resolve_provider_config,
)

# Variables que podrían contaminar los tests desde el entorno real o el .env.
ENV_VARS_TO_CLEAR = (
    "AGENT_PROVIDER",
    "AGENT_MODEL",
    "AGENT_BASE_URL",
    "AGENT_API_KEY",
    "AGENT_WORKDIR",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_API_KEY",
)


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    for variable in ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(variable, raising=False)


def make_settings(**overrides) -> Settings:
    # _env_file=None: aísla los tests del .env real del repositorio.
    return Settings(_env_file=None, **overrides)


class TestPresets:
    def test_openrouter_por_defecto(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        config = resolve_provider_config(make_settings())

        assert config.provider == "openrouter"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "openrouter/free"
        assert config.api_key == "sk-or-test"

    def test_ollama_no_requiere_clave(self):
        config = resolve_provider_config(make_settings(provider="ollama"))

        assert config.base_url == "http://localhost:11434/v1"
        assert config.model == "qwen3:8b"
        assert config.api_key  # valor dummy no vacío: el SDK lo exige

    def test_sobrescrituras_explicitas_ganan(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "legacy")

        config = resolve_provider_config(
            make_settings(
                provider="openrouter",
                model="mi/modelo",
                base_url="https://ejemplo.test/v1",
                api_key="directa",
            )
        )

        assert config.model == "mi/modelo"
        assert config.base_url == "https://ejemplo.test/v1"
        assert config.api_key == "directa"  # AGENT_API_KEY gana sobre la legacy

    def test_clave_legacy_se_usa_si_no_hay_agent_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-legacy")

        config = resolve_provider_config(make_settings(provider="openai"))

        assert config.api_key == "sk-legacy"


class TestErroresAccionables:
    def test_proveedor_desconocido(self):
        with pytest.raises(ConfigError) as info:
            resolve_provider_config(make_settings(provider="inexistente"))
        assert "Proveedor desconocido" in str(info.value)
        assert "ollama" in str(info.value)  # sugiere los disponibles

    def test_falta_clave_en_openrouter(self):
        with pytest.raises(ConfigError) as info:
            resolve_provider_config(make_settings(provider="openrouter"))
        mensaje = str(info.value)
        assert "OPENROUTER_API_KEY" in mensaje
        assert ".env" in mensaje

    def test_custom_requiere_base_url(self):
        with pytest.raises(ConfigError) as info:
            resolve_provider_config(make_settings(provider="custom", model="m"))
        assert "AGENT_BASE_URL" in str(info.value)

    def test_custom_requiere_modelo(self):
        with pytest.raises(ConfigError) as info:
            resolve_provider_config(
                make_settings(provider="custom", base_url="https://x.test/v1")
            )
        assert "AGENT_MODEL" in str(info.value)


class TestSeguridad:
    def test_la_clave_no_aparece_en_repr(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secreta")

        config = resolve_provider_config(make_settings())

        assert "sk-or-secreta" not in repr(config)
