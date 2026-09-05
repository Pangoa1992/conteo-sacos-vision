"""
Módulo de captura de video/imágenes.

Soporta TRES tipos de fuente, para poder empezar a probar el sistema
sin depender de tener la cámara IP de la fábrica desde el día uno:

  1. Carpeta de fotos sueltas   -> ej. "./fotos_prueba/"
  2. Video ya grabado           -> ej. "./videos_prueba/muestra.mp4"
  3. Cámara en vivo (IP o USB)  -> ej. "rtsp://192.168.1.50/stream" o "0" (webcam)

El resto del pipeline (deteccion.py, main.py) no necesita saber cuál
de las tres se está usando: siempre recibe frames de la misma forma.
"""

import os
import glob
import cv2


EXTENSIONES_IMAGEN = (".jpg", ".jpeg", ".png", ".bmp")


def obtener_fuente_video() -> str:
    """
    Devuelve la fuente configurada en la variable de entorno VIDEO_SOURCE.
    Puede ser una carpeta de fotos, un archivo de video, una URL de cámara
    IP (RTSP/HTTP), o "0" para la webcam local.
    """
    return os.getenv("VIDEO_SOURCE", "0")


def tipo_de_fuente(fuente: str) -> str:
    """
    Determina automáticamente qué tipo de fuente es, para que main.py
    sepa cómo leerla.

    Devuelve: "imagenes" | "video" | "camara"
    """
    if os.path.isdir(fuente):
        return "imagenes"

    if os.path.isfile(fuente):
        return "video"

    # No es ni carpeta ni archivo local -> se asume cámara en vivo
    # (RTSP/HTTP de una cámara IP, o "0"/"1" para webcam USB)
    return "camara"


def abrir_captura(fuente: str) -> cv2.VideoCapture:
    """
    Abre un video o una cámara en vivo. (Para el caso de "imágenes",
    se usa `leer_frames_imagenes()` en su lugar, ver más abajo).
    """
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {fuente}. "
            "Verifica la ruta del archivo o que la cámara IP esté encendida y accesible."
        )
    return cap


def leer_frames(cap: cv2.VideoCapture):
    """Generador que entrega los frames de un video/cámara, uno por uno."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def leer_frames_imagenes(carpeta: str):
    """
    Generador que entrega, una por una, las imágenes de una carpeta
    (ordenadas por nombre), simulando "frames" para que el resto del
    pipeline las procese exactamente igual que si vinieran de un video.
    """
    rutas = sorted(
        p for p in glob.glob(os.path.join(carpeta, "*"))
        if p.lower().endswith(EXTENSIONES_IMAGEN)
    )

    if not rutas:
        raise RuntimeError(
            f"La carpeta '{carpeta}' no tiene imágenes ({', '.join(EXTENSIONES_IMAGEN)})."
        )

    for ruta in rutas:
        frame = cv2.imread(ruta)
        if frame is None:
            print(f"[captura] No se pudo leer la imagen, se omite: {ruta}")
            continue
        yield ruta, frame

