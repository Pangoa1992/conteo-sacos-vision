"""
Módulo de detección y conteo de sacos.

ESTADO ACTUAL: implementación simulada (stub). Cuenta un valor
aproximado basado en contornos detectados por color/forma como
placeholder, SOLO para poder probar el pipeline completo
(captura -> detección -> envío al backend) mientras se entrena
o adapta el modelo YOLOv8 real con video/fotos reales de la fábrica.

Cuando se tenga el modelo entrenado, reemplazar `contar_sacos()`
por la inferencia real de YOLOv8, manteniendo la misma firma de
función para no romper el resto del pipeline.
"""

from dataclasses import dataclass
import random


@dataclass
class ResultadoConteo:
    cantidad_sacos: int
    confianza_promedio: float  # 0-100


def cargar_modelo(ruta_modelo: str | None = None):
    """
    Punto de extensión: aquí se cargará el modelo YOLOv8 entrenado,
    por ejemplo:

        from ultralytics import YOLO
        return YOLO(ruta_modelo)

    Por ahora devuelve None, ya que `contar_sacos()` no lo necesita
    todavía (usa una simulación).
    """
    return None


def contar_sacos(frame, modelo=None) -> ResultadoConteo:
    """
    Cuenta los sacos visibles en el frame.

    TODO (cuando haya modelo real):
        resultados = modelo.predict(frame)
        cantidad = len(resultados[0].boxes)
        confianza = float(resultados[0].boxes.conf.mean() * 100)
        return ResultadoConteo(cantidad, confianza)

    Por ahora: valor simulado, solo para probar el flujo end-to-end.
    """
    cantidad_simulada = random.randint(90, 130)
    confianza_simulada = round(random.uniform(90, 98), 1)
    return ResultadoConteo(cantidad_simulada, confianza_simulada)
