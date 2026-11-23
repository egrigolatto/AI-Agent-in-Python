import argparse  # Permite leer argumentos escritos al ejecutar el programa.
import os  # Permite acceder a las variables de entorno.

from dotenv import load_dotenv  # Carga variables definidas en el archivo .env.
from openai import OpenAI  # Proporciona el cliente para llamar a la API.


def main() -> None:  # Punto de entrada con la lógica principal del chatbot.
    parser = argparse.ArgumentParser(description="Chatbot")  # Crea el analizador de argumentos.
    parser.add_argument("user_prompt", type=str, help="User prompt")  # Declara el texto obligatorio del usuario.
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")  # Activa una opción booleana de salida detallada.
    args = parser.parse_args()  # Lee los argumentos recibidos desde la terminal.
    load_dotenv()  # Carga las variables del archivo .env al entorno del proceso.
    api_key = os.environ.get("OPENROUTER_API_KEY")  # Obtiene la clave de OpenRouter.
    if api_key is None:  # Comprueba que la variable de entorno exista.
        raise RuntimeError(
            # Informa cómo configurar la clave si no está definida.
            "La variable de entorno OPENROUTER_API_KEY no está definida. "
            "Configúrala en el entorno o en un archivo .env."
        )

    messages = [
    {"role": "user", "content": args.user_prompt},
    ]

    client = OpenAI(  # Crea el cliente del SDK configurado para OpenRouter.
        base_url="https://openrouter.ai/api/v1",  # Indica la dirección de la API.
        api_key=api_key,  # Proporciona la clave para autenticar las solicitudes.
    )
    response = client.chat.completions.create(  # Envía el mensaje y recibe la respuesta.
        model="openrouter/free",  # Selecciona el modelo o ruta gratuita de OpenRouter.
        messages=messages,  # Contiene los mensajes que forman la conversación.
    )
    
    if response.usage is None:  # Comprueba que la API haya devuelto el consumo de tokens.
        raise RuntimeError("La respuesta de la API no incluyó información de uso de tokens.")  # Indica que falta ese dato.

    if args.verbose:  # Muestra los metadatos solo si se indicó --verbose.
        print(f"User prompt: {args.user_prompt}")  # Muestra el texto ingresado por el usuario.
        print(f"Prompt tokens: {response.usage.prompt_tokens}")  # Muestra los tokens usados en la entrada.
        print(f"Response tokens: {response.usage.completion_tokens}")  # Muestra los tokens generados en la respuesta.
    print(response.choices[0].message.content)  # Muestra el texto de la primera respuesta del modelo.


if __name__ == "__main__":  # Comprueba si este archivo se ejecutó directamente.
    main()  # Ejecuta el chatbot solo cuando se inicia este archivo.
