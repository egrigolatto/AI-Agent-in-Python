"""Pruebas del registro de herramientas: esquemas, inyección y despacho seguro."""

from ai_agent.tools.registry import Tool, ToolContext, ToolRegistry


def _echo(working_directory: str, text: str) -> str:
    return f"{text} en {working_directory}"


def _limited(working_directory: str, max_file_chars: int) -> str:
    return f"{working_directory}:{max_file_chars}"


def _boom(value: int) -> str:
    raise RuntimeError(f"kaboom {value}")


def _plain(text: str) -> str:
    return f"plain {text}"


def make_registry() -> ToolRegistry:
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    return ToolRegistry(
        [
            Tool(name="echo", description="d", parameters=schema, handler=_echo),
            Tool(name="limited", description="d", parameters=schema, handler=_limited),
            Tool(name="boom", description="d", parameters=schema, handler=_boom),
            Tool(name="plain", description="d", parameters=schema, handler=_plain),
        ]
    )


CONTEXT = ToolContext(working_directory="/sandbox", max_file_chars=777)


class TestSchemas:
    def test_formato_openai(self):
        schema = make_registry().get("echo").to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["parameters"]["required"] == ["text"]

    def test_schemas_cubre_todas_las_herramientas(self):
        registry = make_registry()
        assert len(registry.schemas()) == len(registry)
        assert set(registry.names()) == {"boom", "echo", "limited", "plain"}

    def test_registro_duplicado_falla(self):
        registry = make_registry()
        tool = Tool(name="echo", description="x", parameters={}, handler=_plain)
        try:
            registry.register(tool)
            raise AssertionError("debió lanzar ValueError")
        except ValueError as error:
            assert "duplicada" in str(error)


class TestInyeccionDeContexto:
    def test_inyecta_working_directory_por_firma(self):
        result = make_registry().call("echo", {"text": "hi"}, CONTEXT)
        assert result == "hi en /sandbox"

    def test_inyecta_ambos_campos_si_el_handler_los_declara(self):
        result = make_registry().call("limited", {}, CONTEXT)
        assert result == "/sandbox:777"

    def test_el_contexto_sobrescribe_valores_del_modelo(self):
        # Seguridad RF-06: aunque el modelo envíe working_directory, gana el contexto.
        result = make_registry().call(
            "echo", {"text": "hi", "working_directory": "/etc"}, CONTEXT
        )
        assert result == "hi en /sandbox"

    def test_no_inyecta_en_handlers_que_no_lo_declaran(self):
        result = make_registry().call("plain", {"text": "x"}, CONTEXT)
        assert result == "plain x"


class TestDespachoSeguro:
    def test_herramienta_desconocida(self):
        result = make_registry().call("nope", {}, CONTEXT)
        assert "Unknown tool 'nope'" in result
        assert "echo" in result  # lista las disponibles

    def test_argumentos_invalidos_se_convierten_en_error(self):
        result = make_registry().call("echo", {}, CONTEXT)  # falta 'text'
        assert result.startswith("Error: Invalid arguments")

    def test_excepcion_del_handler_se_captura(self):
        result = make_registry().call("boom", {"value": 1}, CONTEXT)
        assert result.startswith("Error: Tool 'boom' failed")
        assert "kaboom" in result


def _cuarenta_y_dos() -> str:  # type: ignore[return-value]
    return 42  # noqa: RET504 - deliberado: el registro debe tolerar handlers torcidos


def test_resultado_no_textual_se_convierte_en_error():
    registry = ToolRegistry(
        [Tool(name="roto", description="d", parameters={}, handler=_cuarenta_y_dos)]
    )
    result = registry.call("roto", {}, CONTEXT)
    assert result.startswith("Error:")
    assert "non-text value" in result
