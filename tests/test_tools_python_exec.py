"""Pruebas de la herramienta `run_python_file`.

Migran y formalizan el script manual `test_run_python_file.py` original.
Usan `sys.executable` implícitamente (la herramienta lo usa, D-08), por lo
que funcionan en cualquier entorno sin depender de un `python` en PATH.
"""

from ai_agent.tools.python_exec import run_python_file


class TestRunPythonFile:
    def test_ejecuta_y_devuelve_stdout(self, tmp_path):
        (tmp_path / "hello.py").write_text(
            "import sys\nprint('hi', sys.argv[1])\n", encoding="utf-8"
        )

        result = run_python_file(str(tmp_path), "hello.py", ["world"])

        assert "STDOUT:\nhi world" in result

    def test_reporta_codigo_de_salida_y_stderr(self, tmp_path):
        (tmp_path / "bad.py").write_text(
            "import sys\nprint('oops', file=sys.stderr)\nsys.exit(3)\n",
            encoding="utf-8",
        )

        result = run_python_file(str(tmp_path), "bad.py")

        assert "Process exited with code 3" in result
        assert "STDERR:\noops" in result

    def test_sin_salida(self, tmp_path):
        (tmp_path / "quiet.py").write_text("pass\n", encoding="utf-8")
        assert "No output produced" in run_python_file(str(tmp_path), "quiet.py")

    def test_timeout(self, tmp_path):
        (tmp_path / "loop.py").write_text(
            "import time\ntime.sleep(10)\n", encoding="utf-8"
        )

        result = run_python_file(str(tmp_path), "loop.py", timeout_seconds=1)

        assert "timed out after 1 seconds" in result

    def test_rechaza_archivos_no_python(self, tmp_path):
        (tmp_path / "datos.txt").write_text("x", encoding="utf-8")
        result = run_python_file(str(tmp_path), "datos.txt")
        assert "is not a Python file" in result

    def test_rechaza_escape_del_sandbox(self, tmp_path):
        resultado = run_python_file(str(tmp_path), "../main.py")
        assert "outside the permitted working directory" in resultado

    def test_archivo_inexistente(self, tmp_path):
        result = run_python_file(str(tmp_path), "nope.py")
        assert "does not exist or is not a regular file" in result

    def test_el_cwd_es_el_directorio_de_trabajo(self, tmp_path):
        (tmp_path / "where.py").write_text(
            "import os\nprint(os.path.basename(os.getcwd()))\n", encoding="utf-8"
        )

        result = run_python_file(str(tmp_path), "where.py")

        assert tmp_path.name in result
