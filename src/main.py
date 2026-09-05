"""
Punto de entrada del módulo de visión computarizada.

Flujo:
  1. Detecta automáticamente el tipo de fuente configurada (fotos, video o cámara en vivo).
  2. Por cada imagen/frame, ejecuta la detección/conteo de sacos.
  3. Envía el resultado al backend web mediante la API REST.

Uso:
    python -m src.main

Configura VIDEO_SOURCE en tu .env con una de estas opciones:
    ./fotos_prueba/              -> carpeta de fotos (una por conteo)
    ./videos_prueba/muestra.mp4  -> video ya grabado
    rtsp://192.168.1.50/stream   -> cámara IP en vivo
    0                             -> webcam de la computadora (pruebas rápidas)
"""

import time
from dotenv import load_dotenv

from .captura import obtener_fuente_video, tipo_de_fuente, abrir_captura, leer_frames, leer_frames_imagenes
from .deteccion import cargar_modelo, contar_sacos
from .api_cliente import enviar_conteo

load_dotenv()

# Cada cuántos frames se hace un conteo y se envía al backend
# (solo aplica a video/cámara; en modo "imágenes" se cuenta cada foto).
INTERVALO_FRAMES = 30


def ejecutar():
    fuente = obtener_fuente_video()
    tipo = tipo_de_fuente(fuente)
    print(f"[main] Fuente detectada: '{fuente}' (tipo: {tipo})")

    modelo = cargar_modelo()

    if tipo == "imagenes":
        _procesar_imagenes(fuente, modelo)
    else:
        _procesar_video_o_camara(fuente, modelo)

    print("[main] Proceso finalizado.")


def _procesar_imagenes(carpeta: str, modelo):
    """Cuenta sacos en cada foto de la carpeta, una por una."""
    for ruta, frame in leer_frames_imagenes(carpeta):
        resultado = contar_sacos(frame, modelo)
        print(f"[main] {ruta} -> {resultado.cantidad_sacos} sacos "
              f"(confianza {resultado.confianza_promedio}%)")

        respuesta = enviar_conteo(resultado, fuente_camara=f"foto:{ruta}")
        print(f"[main] Respuesta del backend: {respuesta}")
        time.sleep(1)


def _procesar_video_o_camara(fuente: str, modelo):
    """Cuenta sacos periódicamente mientras lee un video o una cámara en vivo."""
    cap = abrir_captura(fuente)
    contador_frames = 0

    for frame in leer_frames(cap):
        contador_frames += 1

        if contador_frames % INTERVALO_FRAMES == 0:
            resultado = contar_sacos(frame, modelo)
            print(f"[main] Conteo detectado: {resultado.cantidad_sacos} sacos "
                  f"(confianza {resultado.confianza_promedio}%)")

            respuesta = enviar_conteo(resultado, fuente_camara=fuente)
            print(f"[main] Respuesta del backend: {respuesta}")
            time.sleep(1)


if __name__ == "__main__":
    ejecutar()