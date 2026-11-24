"""Conjunto de herramientas por defecto del agente.

`DEFAULT_TOOLS` es el catálogo que se ofrece al modelo; cada herramienta está
documentada en `docs/tools.md` (qué hace, cómo funciona, por qué existe).
Para agregar una herramienta nueva: crear el handler + `Tool` en un módulo de
este paquete, añadirla a `DEFAULT_TOOLS` y documentarla en `docs/tools.md`.
"""

from ai_agent.tools.files import (
    get_file_content,
    get_files_info,
    tool_get_file_content,
    tool_get_files_info,
    tool_write_file,
    write_file,
)
from ai_agent.tools.python_exec import run_python_file, tool_run_python_file
from ai_agent.tools.registry import Tool, ToolContext, ToolRegistry

DEFAULT_TOOLS: tuple[Tool, ...] = (
    tool_get_files_info,
    tool_get_file_content,
    tool_run_python_file,
    tool_write_file,
)


def build_default_registry() -> ToolRegistry:
    """Construye el registro con las herramientas por defecto."""
    return ToolRegistry(DEFAULT_TOOLS)


__all__ = [
    "DEFAULT_TOOLS",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "build_default_registry",
    "get_file_content",
    "get_files_info",
    "run_python_file",
    "tool_get_file_content",
    "tool_get_files_info",
    "tool_run_python_file",
    "tool_write_file",
    "write_file",
]
