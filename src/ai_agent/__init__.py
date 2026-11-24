"""Agente de IA con herramientas y soporte multi-proveedor.

Paquete principal del proyecto. El núcleo vive en `ai_agent.agent` (clase
`Agent`); los proveedores en `ai_agent.providers`; las herramientas en
`ai_agent.tools`. Ver `README.md` y `docs/` en la raíz del repositorio para
la documentación conceptual y el catálogo de herramientas.
"""

from ai_agent.agent import Agent
from ai_agent.config import ConfigError, Settings, resolve_provider_config
from ai_agent.providers.base import ChatMessage, Completion, LLMProvider, ToolCall, Usage

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "ChatMessage",
    "Completion",
    "ConfigError",
    "LLMProvider",
    "Settings",
    "ToolCall",
    "Usage",
    "__version__",
    "resolve_provider_config",
]
