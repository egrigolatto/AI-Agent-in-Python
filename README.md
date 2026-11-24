# AI Agent in Python

Agente de codificacion por linea de comandos construido con Python, el SDK de OpenAI y la API compatible de OpenRouter. El modelo puede solicitar herramientas para explorar, leer, escribir y ejecutar archivos dentro del proyecto de ejemplo `calculator`.

## Funcionalidades

- Mantiene el historial de mensajes y repite el ciclo de solicitud al modelo despues de cada herramienta, hasta recibir una respuesta final.
- Limita el ciclo a 20 iteraciones.
- Ofrece cuatro herramientas: listar directorios, leer archivos, ejecutar archivos Python y escribir archivos.
- Restringe las rutas de las herramientas al directorio `calculator`.
- Limita la lectura de archivos a `MAX_CHARS` caracteres (10 000 por defecto).
- Captura la salida estandar y los errores de los programas ejecutados, con un limite de 30 segundos.
- Admite `--verbose` para mostrar el prompt, el uso de tokens, las llamadas a herramientas y sus resultados.

El modelo configurado es `openrouter/free`; la disponibilidad y el comportamiento de los modelos gratuitos pueden variar.

## Requisitos

- Python 3.13 o superior.
- [uv](https://docs.astral.sh/uv/).
- Una clave de API de OpenRouter.

## Configuracion

Instala las dependencias desde la raiz del proyecto:

```bash
uv sync
```

Crea un archivo `.env` en la raiz con tu clave:

```dotenv
OPENROUTER_API_KEY=tu_clave_de_openrouter
```

`.env` esta excluido de Git. No publiques ni incluyas tu clave en el repositorio.

## Uso

Ejecuta el agente desde la raiz del repositorio. El prompt debe ir entre comillas si contiene espacios:

```bash
uv run main.py "List the files in the pkg directory"
```

Para ver las llamadas a herramientas y otros detalles:

```bash
uv run main.py "Explain how the calculator renders results" --verbose
```

El agente puede decidir llamar a una o varias herramientas. La carpeta de trabajo se establece en `./calculator`; las rutas que el modelo proporciona son relativas a ella.

## Herramientas disponibles

| Funcion | Uso |
| --- | --- |
| `get_files_info` | Lista los elementos de un directorio con su tamano y si son directorios. |
| `get_file_content` | Lee el contenido de un archivo, hasta el limite configurado. |
| `run_python_file` | Ejecuta un archivo `.py` y devuelve su salida; acepta argumentos opcionales. |
| `write_file` | Crea las carpetas padre necesarias y escribe o sobrescribe un archivo. |

Las declaraciones de herramientas estan en cada modulo de `functions/`; [call_function.py](./call_function.py) las reune y distribuye cada llamada a su funcion. [main.py](./main.py) maneja el ciclo de conversacion y devuelve los resultados de herramientas al modelo en la siguiente iteracion.

## Pruebas manuales

Cada script de prueba imprime los resultados de una herramienta:

```bash
uv run test_get_files_info.py
uv run test_get_file_content.py
uv run test_run_python_file.py
uv run test_write_file.py
uv run calculator/tests.py
```

**Nota:** `test_write_file.py` sobrescribe `calculator/lorem.txt` con texto corto y crea `calculator/pkg/morelorem.txt`. Por eso, para probar el truncamiento con `test_get_file_content.py`, asegúrate de que `calculator/lorem.txt` tenga mas de 10 000 caracteres antes de ejecutar ese test.

## Estructura del proyecto

```text
.
├── main.py
├── prompts.py
├── call_function.py
├── config.py
├── functions/
│   ├── get_files_info.py
│   ├── get_file_content.py
│   ├── run_python_file.py
│   └── write_file.py
├── calculator/
│   ├── main.py
│   ├── tests.py
│   └── pkg/
│       ├── calculator.py
│       └── render.py
├── test_get_files_info.py
├── test_get_file_content.py
├── test_run_python_file.py
├── test_write_file.py
├── pyproject.toml
└── uv.lock
```

- `main.py`: interfaz CLI, solicitud al modelo y ciclo de herramientas.
- `prompts.py`: instrucciones del sistema para el agente.
- `call_function.py`: esquemas disponibles, mapa de funciones y despacho de llamadas.
- `config.py`: limite de caracteres para lectura de archivos.
- `functions/`: implementacion de las herramientas.
- `calculator/`: proyecto de ejemplo sobre el que trabaja el agente.

## Limites de seguridad

La validacion de rutas ayuda a mantener las herramientas dentro de `calculator`, pero no constituye un sandbox completo del sistema operativo. En particular, el agente puede ejecutar codigo Python con los permisos del usuario actual. Utiliza el agente solo con proyectos y archivos de confianza.