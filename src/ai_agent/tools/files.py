"""Herramientas de archivos: listar, leer y escribir dentro del sandbox.

Qué resuelven: son los "ojos y manos" del agente sobre el sistema de
archivos. Con ellas puede explorar un proyecto (`get_files_info`), entender
su código (`get_file_content`) y modificarlo (`write_file`). Sin estas tres,
el modelo solo podría conversar; con ellas puede trabajar sobre archivos
reales.

Cómo funciona el sandbox (RF-06): las tres herramientas resuelven la ruta
relativa contra `working_directory` con `os.path.normpath` (que colapsa
segmentos `..`) y verifican con `os.path.commonpath` que el resultado quede
dentro del directorio de trabajo. Si la ruta escapa (p. ej. `../secreto` o
`/etc/passwd`), la herramienta devuelve un error sin tocar el sistema de
archivos. Límite conocido: es una frontera lógica, no aislamiento de sistema
operativo (ver `docs/technical-design.md`).

`working_directory` y `max_file_chars` nunca los aporta el modelo: los
inyecta el `ToolRegistry` desde el `ToolContext` (ver `registry.py`).
"""

import os

from ai_agent.config import DEFAULT_MAX_FILE_CHARS
from ai_agent.tools.registry import Tool


def _resolve_in_sandbox(
    working_directory: str, relative_path: str
) -> tuple[str | None, str | None]:
    """Resuelve una ruta relativa dentro del directorio de trabajo.

    Devuelve `(ruta_absoluta, None)` si la ruta queda dentro del sandbox, o
    `(None, mensaje_de_error)` si queda fuera. Doble defensa contra path
    traversal:

    1. `os.path.normpath` colapsa segmentos `..` antes de comparar, para que
       `sub/../secreto` no se cuele como ruta "interna".
    2. `os.path.realpath` resuelve symlinks contra la raíz ya resuelta: sin
       esto, un enlace simbólico DENTRO del sandbox que apunte fuera (p. ej.
       `link -> /etc`) pasaría la comprobación sintáctica de `commonpath`.
    """
    working_dir_abs = os.path.realpath(working_directory)
    target = os.path.normpath(os.path.join(working_dir_abs, relative_path))
    target = os.path.realpath(target)
    if os.path.commonpath([working_dir_abs, target]) != working_dir_abs:
        return None, (
            f'Error: Cannot access "{relative_path}" as it is outside the '
            "permitted working directory"
        )
    return target, None


def get_files_info(working_directory: str, directory: str = ".") -> str:
    """Lista los elementos de un directorio con tamaño y tipo.

    Por qué existe: es normalmente la primera herramienta que usa el agente
    para orientarse en un proyecto desconocido (equivalente a `ls -l`).
    Ordena las entradas para que la salida sea estable entre llamadas.
    """
    try:
        target_dir, sandbox_error = _resolve_in_sandbox(working_directory, directory)
        if sandbox_error:
            return sandbox_error
        if not os.path.isdir(target_dir):
            return f'Error: "{directory}" is not a directory'

        lines = []
        for item_name in sorted(os.listdir(target_dir)):
            item_path = os.path.join(target_dir, item_name)
            lines.append(
                f"- {item_name}: file_size={os.path.getsize(item_path)} bytes, "
                f"is_dir={os.path.isdir(item_path)}"
            )
        return "\n".join(lines) if lines else "(empty directory)"
    except (OSError, TypeError, ValueError) as error:
        return f"Error: {error}"


def get_file_content(
    working_directory: str,
    file_path: str,
    max_file_chars: int = DEFAULT_MAX_FILE_CHARS,
) -> str:
    """Lee el contenido de un archivo de texto, con truncamiento.

    Por qué existe: permite al agente leer código o datos antes de
    modificarlos o responder sobre ellos.

    Cómo funciona el truncamiento (RNF-03): lee como máximo `max_file_chars`
    caracteres y agrega una marca si el archivo continúa. Sin este límite, un
    archivo grande desbordaría la ventana de contexto del modelo y
    encarecería cada llamada.
    """
    try:
        target_file, sandbox_error = _resolve_in_sandbox(working_directory, file_path)
        if sandbox_error:
            return sandbox_error
        if not os.path.isfile(target_file):
            return f'Error: File not found or is not a regular file: "{file_path}"'

        with open(target_file, "r", encoding="utf-8") as file:
            content = file.read(max_file_chars)
            if file.read(1):  # ¿queda contenido sin leer?
                content += (
                    f'\n[...File "{file_path}" truncated at {max_file_chars} characters]'
                )
        return content
    except (OSError, TypeError, ValueError) as error:
        # ValueError también cubre UnicodeDecodeError (archivos binarios).
        return f"Error: {error}"


def write_file(working_directory: str, file_path: str, content: str) -> str:
    """Escribe o sobrescribe un archivo, creando las carpetas padre.

    Por qué existe: es la herramienta que permite al agente producir cambios
    reales (crear o editar archivos). Es destructiva —sobrescribe sin pedir
    diff— por eso requiere confirmación del usuario (D-03).
    """
    try:
        target_file, sandbox_error = _resolve_in_sandbox(working_directory, file_path)
        if sandbox_error:
            return sandbox_error
        if os.path.isdir(target_file):
            return f'Error: Cannot write to "{file_path}" as it is a directory'

        parent_directory = os.path.dirname(target_file)
        os.makedirs(parent_directory, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as file:
            file.write(content)
        return (
            f'Successfully wrote to "{file_path}" ({len(content)} characters written)'
        )
    except (OSError, TypeError, ValueError) as error:
        return f"Error: {error}"


# --- Declaraciones de herramientas (esquema + handler + peligrosidad) -------

tool_get_files_info = Tool(
    name="get_files_info",
    description=(
        "Lists files in a specified directory relative to the working "
        "directory, providing file size and directory status."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "Directory path to list files from, relative to the working "
                    "directory (default is the working directory itself)."
                ),
            },
        },
    },
    handler=get_files_info,
)

tool_get_file_content = Tool(
    name="get_file_content",
    description="Reads the contents of a file relative to the working directory.",
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path of the file to read.",
            },
        },
        "required": ["file_path"],
    },
    handler=get_file_content,
)

tool_write_file = Tool(
    name="write_file",
    description=(
        "Writes or overwrites a file relative to the working directory, "
        "creating parent directories as needed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Relative path of the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    },
    handler=write_file,
    requires_confirmation=True,  # destructiva: human-in-the-loop (D-03)
)
