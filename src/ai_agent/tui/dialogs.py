"""Diálogo modal de confirmación de herramientas peligrosas (D-03).

Qué resuelve: el agente necesita una respuesta humana (aprobar/rechazar) a
mitad de su ciclo. La TUI lo resuelve con un `ModalScreen[bool]`: el worker
del turno llama a `push_screen_wait(dialogo)` y queda suspendido hasta que
el usuario pulsa un botón o una tecla, momento en que `dismiss(bool)`
devuelve el valor al worker. Es el patrón bidireccional estándar de textual
y encaja con el `confirm_handler` inyectable del núcleo sin modificarlo.
"""

import json

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from ai_agent.agent import ConfirmationRequest

MAX_ARGUMENTS_CHARS = 1200


class ConfirmationDialog(ModalScreen[bool]):
    """Pregunta al usuario si aprueba ejecutar una herramienta peligrosa."""

    DEFAULT_CSS = """
    ConfirmationDialog {
        align: center middle;
    }
    ConfirmationDialog > Vertical {
        width: 90;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $warning;
        padding: 1 2;
    }
    ConfirmationDialog .args {
        color: $text-muted;
        height: auto;
        max-height: 14;
        overflow-y: auto;
        margin: 1 0;
    }
    ConfirmationDialog .buttons {
        height: auto;
        align-horizontal: center;
        margin-top: 1;
    }
    ConfirmationDialog .buttons Button {
        margin: 0 2;
    }
    """

    BINDINGS = [
        Binding("s", "approve", "Aprobar", show=False),
        Binding("n", "deny", "Rechazar", show=False),
        Binding("escape", "deny", "Rechazar", show=False),
    ]

    def __init__(self, request: ConfirmationRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self):
        arguments = json.dumps(self.request.arguments, ensure_ascii=False, indent=2)
        if len(arguments) > MAX_ARGUMENTS_CHARS:
            arguments = arguments[:MAX_ARGUMENTS_CHARS] + "\n…"
        with Vertical():
            # markup=False: los argumentos vienen del modelo (texto no confiable).
            yield Static(
                f"El agente quiere ejecutar la herramienta '{self.request.tool_name}' "
                "con estos argumentos:",
                markup=False,
            )
            yield Static(arguments, classes="args", markup=False)
            with Horizontal(classes="buttons"):
                yield Button("Aprobar (s)", id="approve", variant="success")
                yield Button("Rechazar (n)", id="deny", variant="error")

    @on(Button.Pressed)
    def handle_button(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
