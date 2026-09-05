"""
Punto de entrada del módulo de visión computarizada.

Flujo:
  1. Abre la fuente de video (cámara IP real o archivo de prueba).
  2. Por cada lote de frames, ejecuta la detección/conteo de sacos.
  3. Envía el resultado al backend web mediante la API REST.

Uso:
    python -m src.main
"""

import time
from dotenv import load_dotenv

from .captura import obtener_fuente_video, abrir_captura, leer_frames
from .deteccion import cargar_modelo, contar_sacos
from .api_cliente import enviar_conteo

load_dotenv()

# Cada cuántos frames se hace un conteo y se envía al backend.
# (En la implementación real, esto se ajustará según el FPS de la
# cámara y el tiempo que tarda un saco en pasar por el punto de conteo).
INTERVALO_FRAMES = 30


def ejecutar():
    fuente = obtener_fuente_video()
    print(f"[main] Iniciando captura desde: {fuente}")

    cap = abrir_captura(fuente)
    modelo = cargar_modelo()

    contador_frames = 0
    for frame in leer_frames(cap):
        contador_frames += 1

        if contador_frames % INTERVALO_FRAMES == 0:
            resultado = contar_sacos(frame, modelo)
            print(f"[main] Conteo detectado: {resultado.cantidad_sacos} sacos "
                  f"(confianza {resultado.confianza_promedio}%)")

            respuesta = enviar_conteo(resultado)
            print(f"[main] Respuesta del backend: {respuesta}")

            time.sleep(1)  # evita saturar el backend con envíos muy seguidos

    print("[main] Fin del video / stream.")


if __name__ == "__main__":
    ejecutar()
