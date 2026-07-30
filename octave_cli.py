import argparse
import mimetypes
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


def load_config(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"No se encontró el archivo de configuración: {path}")

    config = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()

    for required in ("prompt_file", "output_file"):
        if required not in config:
            sys.exit(f"Falta la clave '{required}' en {path}")

    config.setdefault("model", "gemini-3.1-flash-lite")
    return config


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", text.strip(), flags=re.DOTALL)
    return match.group(1) if match else text


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def find_latest_image(directory: Path) -> Path:
    if not directory.exists():
        sys.exit(f"No se encontró la carpeta de imágenes: {directory}")

    images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        sys.exit(f"No hay imágenes en la carpeta: {directory}")

    return max(images, key=lambda p: p.stat().st_mtime)


def load_image_part(image_path: Path) -> types.Part:
    if not image_path.exists():
        sys.exit(f"No se encontró la imagen: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image/"):
        sys.exit(f"No se pudo determinar un tipo de imagen válido para: {image_path}")

    return types.Part.from_bytes(data=image_path.read_bytes(), mime_type=mime_type)


def main():
    parser = argparse.ArgumentParser(description="CLI para interactuar con la API de Gemini")
    parser.add_argument("--config", default="config.txt", help="Ruta al archivo de configuración")
    parser.add_argument(
        "--imagen",
        action="append",
        default=[],
        dest="imagenes",
        help="Ruta a una imagen para adjuntar. Se puede repetir para enviar varias.",
    )
    parser.add_argument(
        "--sin-imagen",
        action="store_true",
        help="No adjuntar ninguna imagen, aunque haya una carpeta configurada en images_dir.",
    )
    args = parser.parse_args()

    load_dotenv()

    config = load_config(Path(args.config))
    prompt_path = Path(config["prompt_file"])
    output_path = Path(config["output_file"])

    if not prompt_path.exists():
        sys.exit(f"No se encontró el archivo de prompt: {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt_text:
        sys.exit(f"El archivo de prompt está vacío: {prompt_path}")

    image_paths = [Path(imagen) for imagen in args.imagenes]
    if not image_paths and not args.sin_imagen:
        images_dir = config.get("images_dir")
        if images_dir:
            image_paths = [find_latest_image(Path(images_dir))]

    contents = [prompt_text]
    for image_path in image_paths:
        contents.append(load_image_part(image_path))

    generate_config = None
    system_file = config.get("system_file")
    if system_file:
        system_path = Path(system_file)
        if not system_path.exists():
            sys.exit(f"No se encontró el archivo de contexto: {system_path}")
        system_text = system_path.read_text(encoding="utf-8").strip()
        if system_text:
            generate_config = types.GenerateContentConfig(system_instruction=system_text)

    try:
        client = genai.Client()
    except Exception as exc:
        sys.exit(
            "No se pudo inicializar el cliente de Gemini. "
            "Verificá que la variable de entorno GEMINI_API_KEY esté configurada.\n"
            f"Detalle: {exc}"
        )

    try:
        response = client.models.generate_content(
            model=config["model"], contents=contents, config=generate_config
        )
    except Exception as exc:
        sys.exit(f"Error al llamar a la API de Gemini: {exc}")

    output_path.write_text(strip_code_fence(response.text or ""), encoding="utf-8")
    print(f"Respuesta guardada en: {output_path}")


if __name__ == "__main__":
    main()
