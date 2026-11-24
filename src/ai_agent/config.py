"""Configuración del agente: presets de proveedor, entorno y archivo `.env`.

Cómo se resuelve la configuración (de mayor a menor prioridad):

1. Argumentos de CLI (`--provider`, `--model`, `--base-url`, `--workdir`).
2. Variables de entorno / `.env` con prefijo `AGENT_` (`AGENT_PROVIDER`,
   `AGENT_MODEL`, `AGENT_BASE_URL`, `AGENT_API_KEY`, `AGENT_WORKDIR`, ...).
3. Preset del proveedor (base URL y modelo por defecto).
4. Defaults del código (workdir `./calculator`, 20 iteraciones, 10 000 caracteres).

La clave de API se resuelve en cascada (decisión D-07 en `docs/decisions.md`):
`AGENT_API_KEY` → variable legacy del preset (p. ej. `OPENROUTER_API_KEY`, para
que un `.env` existente siga funcionando) → valor por defecto no secreto
(Ollama usa el literal `"ollama"` porque el SDK de OpenAI exige una clave no
vacía, aunque el servidor la ignora).

Por qué pydantic-settings: valida tipos, integra `.env` sin código manual y
genera errores claros; la alternativa (argparse + os.environ a mano) duplica
lógica y no valida.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Defaults compartidos con el resto del paquete (herramientas, agente).
DEFAULT_MAX_FILE_CHARS = 10_000
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_WORKDIR = Path("./calculator")


class ConfigError(RuntimeError):
    """Error de configuración con un mensaje accionable para el usuario (RNF-01)."""


@dataclass(frozen=True)
class ProviderPreset:
    """Valores conocidos de un proveedor compatible con la API de OpenAI.

    - `key_env_vars`: variables de entorno (legacy) donde buscar la clave si no
      se definió `AGENT_API_KEY`.
    - `api_key_required`: si es True y no aparece ninguna clave, el arranque
      falla con un error explicativo en lugar de un fallo críptico de red.
    - `fallback_api_key`: valor no secreto para servidores que ignoran la clave
      (Ollama) pero cuyo SDK la exige no vacía.
    """

    name: str
    base_url: str
    default_model: str
    key_env_vars: tuple[str, ...] = ()
    api_key_required: bool = True
    fallback_api_key: str | None = None


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="openrouter/free",
        key_env_vars=("OPENROUTER_API_KEY",),
    ),
    "openai": ProviderPreset(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        key_env_vars=("OPENAI_API_KEY",),
    ),
    "ollama": ProviderPreset(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen3:8b",
        api_key_required=False,
        fallback_api_key="ollama",
    ),
    "custom": ProviderPreset(
        name="custom",
        base_url="",  # exige AGENT_BASE_URL o --base-url
        default_model="",  # exige AGENT_MODEL o --model
        api_key_required=False,
    ),
}


class Settings(BaseSettings):
    """Ajustes del agente leídos del entorno (prefijo `AGENT_`) y de `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "openrouter"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = Field(default=None, repr=False)  # repr=False: nunca la muestran los logs
    workdir: Path = DEFAULT_WORKDIR
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS
    temperature: float = DEFAULT_TEMPERATURE
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    verbose: bool = False


@dataclass(frozen=True)
class ProviderConfig:
    """Configuración ya resuelta y validada para construir un proveedor."""

    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)  # repr=False: no se filtra al imprimir


def _resolve_api_key(settings: Settings, preset: ProviderPreset) -> str | None:
    """Aplica la cascada de resolución de clave descrita en el docstring."""
    if settings.api_key:
        return settings.api_key
    for env_var in preset.key_env_vars:
        value = os.environ.get(env_var)
        if value:
            return value
    return preset.fallback_api_key


def resolve_provider_config(settings: Settings) -> ProviderConfig:
    """Valida y resuelve la configuración del proveedor activo.

    Qué resuelve: convierte la combinación de preset + entorno + CLI en un
    único `ProviderConfig` listo para el adaptador, o lanza `ConfigError` con
    instrucciones concretas (qué variable definir y en qué archivo).
    """
    preset = PROVIDER_PRESETS.get(settings.provider)
    if preset is None:
        available = ", ".join(sorted(PROVIDER_PRESETS))
        raise ConfigError(
            f"Proveedor desconocido '{settings.provider}'. Disponibles: {available}. "
            "Cámbialo con --provider o AGENT_PROVIDER."
        )

    base_url = settings.base_url or preset.base_url
    if not base_url:
        raise ConfigError(
            "El proveedor 'custom' necesita una base URL: defínela con "
            "AGENT_BASE_URL en el entorno/.env o con --base-url."
        )

    model = settings.model or preset.default_model
    if not model:
        raise ConfigError(
            f"El proveedor '{preset.name}' no tiene modelo por defecto en esta "
            "configuración: define AGENT_MODEL en el entorno/.env o usa --model."
        )

    api_key = _resolve_api_key(settings, preset)
    if api_key is None:
        if preset.api_key_required:
            expected = " o ".join(("AGENT_API_KEY", *preset.key_env_vars))
            raise ConfigError(
                f"No se encontró ninguna clave de API para el proveedor '{preset.name}'. "
                f"Define {expected} en el entorno o en el archivo .env."
            )
        # Servidores compatibles que ignoran la autenticación pero cuyo SDK exige un valor.
        api_key = "not-needed"

    return ProviderConfig(
        provider=preset.name,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
