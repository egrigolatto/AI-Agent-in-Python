import os  # Permite usar funciones para normalizar y comparar rutas.


def get_files_info(working_directory: str, directory: str = ".") -> str:  # Recibe la raíz permitida y una ruta relativa a ella.
    try:  # Agrupa las operaciones que pueden fallar al procesar las rutas.
        working_dir_abs = os.path.abspath(working_directory)  # Convierte la raíz permitida en una ruta absoluta.
        target_dir = os.path.normpath(os.path.join(working_dir_abs, directory))  # Une la raíz con el destino y simplifica segmentos como "..".
        # Comprueba que la ruta común entre la raíz y el destino sea la raíz.
        valid_target_dir = (
            os.path.commonpath([working_dir_abs, target_dir]) == working_dir_abs
        )

        if not valid_target_dir:  # Entra aquí cuando el destino queda fuera de la raíz permitida.
            return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'  # Devuelve el error requerido.

        if not os.path.isdir(target_dir):  # Comprueba que el destino exista y sea un directorio.
            return f'Error: "{directory}" is not a directory'  # Devuelve el error si no es un directorio.

        entries = []  # Guarda los datos de cada elemento encontrado.
        for item_name in os.listdir(target_dir):  # Recorre los nombres dentro del directorio destino.
            item_path = os.path.join(target_dir, item_name)  # Construye la ruta completa de cada elemento.
            entries.append(
                {
                    "name": item_name,  # Registra el nombre del elemento.
                    "file_size": os.path.getsize(item_path),  # Registra su tamaño en bytes.
                    "is_dir": os.path.isdir(item_path),  # Registra si es un directorio.
                }
            )

        formatted_entries = [  # Convierte los datos de cada elemento al formato solicitado.
            f"- {entry['name']}: file_size={entry['file_size']} bytes, is_dir={entry['is_dir']}"
            for entry in entries
        ]
        return "\n".join(formatted_entries)  # Devuelve una línea por elemento del directorio.
    except (OSError, TypeError, ValueError) as error:  # Captura errores esperables de las funciones de rutas.
        return f"Error: {error}"  # Convierte la excepción en un resultado de texto con el prefijo requerido.