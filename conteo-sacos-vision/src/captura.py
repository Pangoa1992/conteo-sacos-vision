"""
Módulo de captura de video.

Hoy: puede leer tanto de un archivo de video local (para desarrollo,
mientras no hay acceso a la cámara IP real de la fábrica) como de un
stream de cámara IP (RTSP/HTTP), sin cambiar el resto del pipeline.
"""

import os
import cv2


def obtener_fuente_video() -> str:
    """
    Devuelve la fuente de video a usar, leída de la variable de entorno
    VIDEO_SOURCE. Puede ser:
      - Ruta a un archivo .mp4 (video ya grabado en la fábrica)
      - URL RTSP/HTTP de una cámara IP real
    """
    return os.getenv("VIDEO_SOURCE", "0")  # "0" = webcam local, útil para probar


def abrir_captura(fuente: str) -> cv2.VideoCapture:
    """
    Abre la fuente de video indicada. Lanza un error claro si no se
    pudo abrir (cámara apagada, archivo no encontrado, etc.), en vez
    de fallar de forma confusa más adelante en el pipeline.
    """
    cap = cv2.VideoCapture(fuente)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {fuente}. "
            "Verifica la ruta del archivo o que la cámara IP esté encendida y accesible."
        )
    return cap


def leer_frames(cap: cv2.VideoCapture):
    """Generador simple que entrega los frames uno por uno."""
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()
