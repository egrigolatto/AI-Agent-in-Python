"""Herramienta de ejecución de archivos Python dentro del sandbox.

Qué resuelve: cierra el ciclo completo del agente (explorar → modificar →
ejecutar → verificar). Sin ella, el modelo no puede comprobar el efecto de
sus cambios; con ella, puede correr los tests o el programa del proyecto y
reaccionar a la salida. Es la herramienta que convierte un "chat que escribe
código" en un agente que itera sobre código real.

Cómo funciona: lanza `sys.executable <archivo> [args...]` como subproceso
con `cwd=working_directory`, captura stdout/stderr por separado, aplica un
timeout (30 s por defecto) y convierte todo (código de salida, salida,
error) en texto para el modelo. Se usa `sys.executable` —el mismo
intérprete que corre el agente— en lugar de `python`, que puede no existir
en el PATH o apuntar a otra versión (decisión D-08).

Seguridad: es la herramienta más peligrosa del conjunto porque ejecuta
código real con los permisos del usuario. El sandbox limita QUÉ archivos
pueden ejecutarse (solo `.py` dentro del directorio de trabajo), pero una
vez lanzado, el proceso no está aislado. Por eso requiere confirmación
humana en cada ejecución (D-03).
"""

import os
import subprocess
import sys

from ai_agent.tools.files import _resolve_in_sandbox
from ai_agent.tools.registry import Tool

DEFAULT_TIMEOUT_SECONDS = 30


def run_python_file(
    working_directory: str,
    file_path: str,
    args: list[str] | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Ejecuta un archivo `.py` del sandbox y devuelve su salida como texto."""
    try:
        absolute_file_path, sandbox_error = _resolve_in_sandbox(
            working_directory, file_path
        )
        if sandbox_error:
            return sandbox_error
        if not os.path.isfile(absolute_file_path):
            return f'Error: "{file_path}" does not exist or is not a regular file'
        if not file_path.endswith(".py"):
            return f'Error: "{file_path}" is not a Python file'

        command = [sys.executable, absolute_file_path]
        if args:
            command.extend(str(arg) for arg in args)

        completed_process = subprocess.run(
            command,
            cwd=os.path.abspath(working_directory),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        output = []
        if completed_process.returncode != 0:
            output.append(f"Process exited with code {completed_process.returncode}")
        if not completed_process.stdout and not completed_process.stderr:
            output.append("No output produced")
        else:
            if completed_process.stdout:
                output.append(f"STDOUT:\n{completed_process.stdout}")
            if completed_process.stderr:
                output.append(f"STDERR:\n{completed_process.stderr}")
        return "\n".join(output)
    except subprocess.TimeoutExpired:
        return (
            f'Error: Execution of "{file_path}" timed out after '
            f"{timeout_seconds} seconds"
        )
    except (OSError, TypeError, ValueError) as error:
        return f"Error: executing Python file: {error}"


tool_run_python_file = Tool(
    name="run_python_file",
    description=(
        "Executes a Python file relative to the working directory and returns "
        "its output."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path of the Python file to execute.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional command-line arguments to pass to the Python file."
                ),
            },
        },
        "required": ["file_path"],
    },
    handler=run_python_file,
    requires_confirmation=True,  # ejecuta código real: human-in-the-loop (D-03)
)
