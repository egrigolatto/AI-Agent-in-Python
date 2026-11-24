"""Pruebas de las herramientas de archivos (sandbox, truncamiento, escritura).

Migran y formalizan los casos que cubrían los scripts manuales
`test_get_files_info.py`, `test_get_file_content.py` y `test_write_file.py`
de la versión original: ahora verifican resultados automáticamente y usan
`tmp_path` en lugar de tocar la carpeta `calculator/`.
"""

import pytest

from ai_agent.tools.files import get_file_content, get_files_info, write_file


class TestGetFilesInfo:
    def test_lista_archivos_y_directorios(self, tmp_path):
        (tmp_path / "a.txt").write_text("hola", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        result = get_files_info(str(tmp_path), ".")

        assert "- a.txt: file_size=4 bytes, is_dir=False" in result
        assert "- sub: file_size=4096 bytes, is_dir=True" in result or (
            "- sub:" in result and "is_dir=True" in result
        )

    def test_directorio_vacio(self, tmp_path):
        assert get_files_info(str(tmp_path), ".") == "(empty directory)"

    def test_rechaza_escape_con_puntos(self, tmp_path):
        result = get_files_info(str(tmp_path), "../")
        assert "outside the permitted working directory" in result

    @pytest.mark.parametrize(
        "ruta_externa",
        [
            "..",
            "../..",
            "./a/../../..",
            "/etc",
            "/etc/passwd",
            "sub/../../..",
            "/proc/self/environ",
        ],
    )
    def test_fuzzing_ligero_de_escape(self, tmp_path, ruta_externa):
        """T5.2: variantes de path traversal deben rechazarse siempre."""
        assert "outside the permitted working directory" in get_files_info(
            str(tmp_path), ruta_externa
        )

    def test_rechaza_ruta_absoluta_externa(self, tmp_path):
        result = get_files_info(str(tmp_path), "/bin")
        assert "outside the permitted working directory" in result

    def test_directorio_inexistente(self, tmp_path):
        result = get_files_info(str(tmp_path), "noexiste")
        assert "is not a directory" in result


class TestGetFileContent:
    def test_lee_contenido_completo(self, tmp_path):
        (tmp_path / "f.txt").write_text("contenido", encoding="utf-8")
        assert get_file_content(str(tmp_path), "f.txt") == "contenido"

    def test_trunca_al_limite(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 150, encoding="utf-8")

        result = get_file_content(str(tmp_path), "big.txt", max_file_chars=100)

        assert result.startswith("x" * 100)
        assert "truncated at 100 characters" in result

    def test_sin_marca_de_truncamiento_si_entra(self, tmp_path):
        (tmp_path / "small.txt").write_text("abc", encoding="utf-8")
        assert "truncated" not in get_file_content(str(tmp_path), "small.txt")

    def test_rechaza_escape(self, tmp_path):
        secreto = tmp_path.parent / "secreto.txt"
        secreto.write_text("no leer", encoding="utf-8")
        result = get_file_content(str(tmp_path), "../secreto.txt")
        assert "outside the permitted working directory" in result

    def test_archivo_inexistente(self, tmp_path):
        result = get_file_content(str(tmp_path), "nope.txt")
        assert "File not found" in result

    def test_directorio_no_es_archivo(self, tmp_path):
        (tmp_path / "d").mkdir()
        result = get_file_content(str(tmp_path), "d")
        assert "File not found or is not a regular file" in result


class TestWriteFile:
    def test_escribe_y_crea_carpetas_padre(self, tmp_path):
        result = write_file(str(tmp_path), "deep/dir/f.txt", "contenido")

        assert result.startswith("Successfully wrote")
        assert (tmp_path / "deep" / "dir" / "f.txt").read_text(encoding="utf-8") == (
            "contenido"
        )

    def test_sobrescribe_archivo_existente(self, tmp_path):
        (tmp_path / "f.txt").write_text("viejo", encoding="utf-8")
        write_file(str(tmp_path), "f.txt", "nuevo")
        assert (tmp_path / "f.txt").read_text(encoding="utf-8") == "nuevo"

    def test_rechaza_escape_y_no_escribe_fuera(self, tmp_path):
        result = write_file(str(tmp_path), "../escape.txt", "peligro")

        assert "outside the permitted working directory" in result
        assert not (tmp_path.parent / "escape.txt").exists()

    def test_rechaza_escribir_sobre_directorio(self, tmp_path):
        (tmp_path / "d").mkdir()
        result = write_file(str(tmp_path), "d", "x")
        assert "is a directory" in result


class TestSymlinks:
    """Defensa adicional: enlaces que apuntan fuera del sandbox (D-15)."""

    def test_lectura_a_traves_de_symlink_externo_se_rechaza(self, tmp_path):
        externo = tmp_path.parent / "secreto-real.txt"
        externo.write_text("no leer", encoding="utf-8")
        (tmp_path / "link.txt").symlink_to(externo)

        result = get_file_content(str(tmp_path), "link.txt")

        assert "outside the permitted working directory" in result

    def test_listado_a_traves_de_symlink_de_directorio_se_rechaza(self, tmp_path):
        (tmp_path / "fuera").mkdir()
        (tmp_path / "enlace").symlink_to(tmp_path / "fuera")

        result = get_files_info(str(tmp_path), "enlace/../../fuera")

        assert "outside the permitted working directory" in result

    def test_symlink_interno_si_se_puede_leer(self, tmp_path):
        (tmp_path / "real.txt").write_text("adentro", encoding="utf-8")
        (tmp_path / "espejo.txt").symlink_to(tmp_path / "real.txt")

        assert get_file_content(str(tmp_path), "espejo.txt") == "adentro"
