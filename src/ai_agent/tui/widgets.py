"""Widgets del chat: burbujas de mensaje y líneas de actividad.

Decisión de render: texto plano con `markup=False` en todos los widgets. El
contenido del modelo y los resultados de herramientas son texto no confiable:
sin `markup=False`, un `Static` interpretaría corchetes al estilo Rich
(`[bold]`, `[red]`) incrustados en la respuesta como formato, rompiendo el
render o permitiendo "inyección de markup".
"""

from textual.widgets import Static


class UserMessage(Static):
    """Burbuja con el mensaje del usuario."""

    def __init__(self, text: str) -> None:
        super().__init__(f"› {text}", classes="user", markup=False)
        self.user_text = text  # texto original, útil para tests y accesibilidad


class AssistantMessage(Static):
    """Burbuja del asistente con soporte de streaming.

    Cómo funciona: mientras el modelo genera, `append_delta` acumula los
    fragmentos de texto (eventos `MessageDelta`) en un búfer y re-renderiza;
    al cerrar el ciclo, `set_final` fija el contenido definitivo. Es un
    `Static` simple a propósito: renderizar Markdown en vivo exige reparsear
    el documento por cada token; queda documentado como mejora futura.
    """

    def __init__(self) -> None:
        super().__init__("", classes="assistant", markup=False)
        self._buffer = ""

    def append_delta(self, text: str) -> None:
        self._buffer += text
        self.update(self._buffer)

    def set_final(self, content: str) -> None:
        self._buffer = content
        self.update(content)


class ToolActivity(Static):
    """Línea secundaria: tool calls, resultados, avisos y errores."""

    def __init__(self, text: str, *, extra_class: str = "") -> None:
        classes = f"tool {extra_class}".strip()
        super().__init__(text, classes=classes, markup=False)
        self.activity_text = text  # texto original, útil para tests y depuración
