"""Registro de herramientas: única fuente de verdad de esquemas y handlers.

Qué resuelve: la versión original del proyecto mantenía una lista de esquemas
y un mapa de funciones duplicados a mano en `call_function.py`; agregar una
herramienta exigía editar dos lugares. Aquí cada herramienta es un objeto
`Tool` autosuficiente (esquema JSON + handler + peligrosidad) y el registro
los reúne (RF-04).

Conceptos clave:

- `Tool.parameters` es el esquema JSON Schema que ve el modelo: es el
  "contrato" con el que el LLM decide qué argumentos enviar.
- `requires_confirmation` activa el patrón human-in-the-loop (D-03): el
  agente consulta a la interfaz antes de ejecutar la herramienta.
- `ToolContext` se inyecta por nombre de parámetro en la firma del handler
  (p. ej. `working_directory`). La inyección la hace el registro, no el
  modelo: por seguridad, el contexto SIEMPRE sobrescribe lo que el modelo
  haya enviado con esos nombres, de modo que el sandbox no lo elige el LLM
  (RF-06).
- `call()` nunca lanza excepciones: todo error se convierte en texto
  `"Error: ..."` que vuelve al modelo como resultado de la herramienta. Así
  un fallo no rompe el ciclo; el modelo puede corregirse y reintentar.
"""

import inspect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

# Campos del contexto que el registro sabe inyectar en los handlers.
INJECTABLE_CONTEXT_FIELDS = ("working_directory", "max_file_chars")


@dataclass(frozen=True)
class ToolContext:
    """Contexto de ejecución inyectado en los handlers que lo declaran."""

    working_directory: str
    max_file_chars: int = 10_000


@dataclass(frozen=True)
class Tool:
    """Definición completa de una herramienta del agente."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]
    requires_confirmation: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Devuelve el schema en el formato `tools` de la API de OpenAI."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Colección de herramientas con despacho seguro por nombre."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Herramienta duplicada: '{tool.name}'")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Esquemas de todas las herramientas, listos para el parámetro `tools`."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def call(self, name: str, arguments: dict[str, Any], context: ToolContext) -> str:
        """Ejecuta una herramienta de forma síncrona y devuelve texto para el modelo.

        Pasos: valida que la herramienta exista → inyecta el contexto según la
        firma del handler → ejecuta → convierte cualquier fallo (argumentos
        inválidos, excepción del handler, resultado no textual) en un mensaje
        `"Error: ..."` que el modelo recibirá como resultado.

        Nota: es síncrona a propósito; el agente la ejecuta con
        `asyncio.to_thread` para no bloquear el event loop (RNF-05).
        """
        tool = self._tools.get(name)
        if tool is None:
            return (
                f"Error: Unknown tool '{name}'. "
                f"Available tools: {', '.join(self.names())}."
            )
        try:
            signature = inspect.signature(tool.handler)
            injected = {
                field_name: getattr(context, field_name)
                for field_name in INJECTABLE_CONTEXT_FIELDS
                if field_name in signature.parameters
            }
            # El contexto va al final: sobrescribe cualquier valor homónimo
            # que haya enviado el modelo (seguridad del sandbox, RF-06).
            kwargs = {**arguments, **injected}
            result = tool.handler(**kwargs)
        except TypeError as error:
            return f"Error: Invalid arguments for tool '{name}': {error}"
        except Exception as error:  # noqa: BLE001 - el ciclo del agente no debe romperse
            return f"Error: Tool '{name}' failed: {error}"
        if not isinstance(result, str):
            return (
                f"Error: Tool '{name}' returned a non-text value "
                f"({type(result).__name__}); tools must return str."
            )
        return result
