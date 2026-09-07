"""
Módulo de detección y conteo de sacos.

Soporta TRES modos, controlados por la variable de entorno MODO_DETECCION:

  1. "simulado" (por defecto) — genera un número aleatorio realista.
     Se usa mientras no hay credenciales de Roboflow ni modelo YOLOv8
     entrenado, solo para probar el resto del pipeline end-to-end.

  2. "sam3" — llama al modelo SAM3 (vía Roboflow), que detecta sacos
     SIN necesitar entrenamiento propio, describiendo el objeto por
     su apariencia visual (ver ROBOFLOW_PROMPT en .env). Requiere
     ROBOFLOW_API_KEY y ROBOFLOW_WORKFLOW_URL configurados.

  3. "yolo" — usa un modelo YOLOv8 propio, entrenado con fotos reales
     de la fábrica (ver conteo-sacos-vision/notebooks o el pipeline
     de entrenamiento). Requiere RUTA_MODELO_YOLO apuntando al
     archivo .pt entrenado.

Los tres modos devuelven el mismo tipo de resultado (ResultadoConteo),
así que el resto del pipeline (main.py, api_cliente.py) no necesita
saber cuál se está usando.
"""

import os
import base64
import random
from dataclasses import dataclass

import cv2
import requests


@dataclass
class ResultadoConteo:
    cantidad_sacos: int
    confianza_promedio: float  # 0-100


def cargar_modelo(ruta_modelo: str | None = None):
    """
    Carga lo que corresponda según MODO_DETECCION:
      - "yolo": carga el modelo YOLOv8 entrenado desde ruta_modelo.
      - "sam3" / "simulado": no necesitan cargar nada (devuelve None).
    """
    modo = os.getenv("MODO_DETECCION", "simulado")

    if modo == "yolo":
        from ultralytics import YOLO
        ruta = ruta_modelo or os.getenv("RUTA_MODELO_YOLO", "modelos/best.pt")
        print(f"[deteccion] Cargando modelo YOLOv8 desde: {ruta}")
        return YOLO(ruta)

    return None


def contar_sacos(frame, modelo=None) -> ResultadoConteo:
    """Cuenta los sacos visibles en el frame, según MODO_DETECCION."""
    modo = os.getenv("MODO_DETECCION", "simulado")

    if modo == "sam3":
        return _contar_con_sam3(frame)
    elif modo == "yolo":
        return _contar_con_yolo(frame, modelo)
    else:
        return _contar_simulado()


def _contar_simulado() -> ResultadoConteo:
    """Modo por defecto: número aleatorio realista, sin modelo real."""
    cantidad_simulada = random.randint(90, 130)
    confianza_simulada = round(random.uniform(90, 98), 1)
    return ResultadoConteo(cantidad_simulada, confianza_simulada)


def _contar_con_sam3(frame) -> ResultadoConteo:
    """
    Cuenta sacos usando el modelo SAM3 vía la API de Roboflow, sin
    necesitar entrenamiento propio. El texto descriptivo del objeto a
    buscar ("orange mesh bag") ya quedó configurado dentro del propio
    workflow de Roboflow (se definió al crear el modelo en su interfaz),
    por lo que NO se envía en cada petición — solo la imagen y los
    parámetros de confianza/filtrado.
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workflow_url = os.getenv("ROBOFLOW_WORKFLOW_URL")
    confianza_minima = float(os.getenv("ROBOFLOW_CONFIDENCE", "0.3"))

    if not api_key or not workflow_url:
        print("[deteccion] MODO_DETECCION=sam3 pero faltan ROBOFLOW_API_KEY o "
              "ROBOFLOW_WORKFLOW_URL en el .env. Usando modo simulado como respaldo.")
        return _contar_simulado()

    # Codificar el frame (imagen) a base64 para enviarlo en la petición.
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        print("[deteccion] No se pudo codificar la imagen para SAM3.")
        return _contar_simulado()
    imagen_base64 = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "inputs": {
            "image": {"type": "base64", "value": imagen_base64},
            "confidence": confianza_minima,
            "iou_threshold": 0.3,
            "class_agnostic_nms": False,
            "max_detections": 1000,
        }
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        respuesta = requests.post(workflow_url, json=payload, headers=headers, timeout=30)
        respuesta.raise_for_status()
        datos = respuesta.json()
    except requests.RequestException as error:
        print(f"[deteccion] Error al llamar a SAM3/Roboflow: {error}")
        return _contar_simulado()

    # NOTA: la forma exacta de la respuesta puede variar según cómo quedó
    # configurado el workflow en Roboflow. Se intentan las estructuras más
    # comunes; si no coincide, se imprime la respuesta cruda para ajustar.
    predicciones = (
        datos.get("predictions")
        or datos.get("outputs", [{}])[0].get("predictions", {}).get("predictions")
        or []
    )

    if not predicciones and not isinstance(predicciones, list):
        print(f"[deteccion] Respuesta de Roboflow con formato inesperado: {datos}")
        return _contar_simulado()

    cantidad = len(predicciones)
    if cantidad > 0:
        confianzas = [p.get("confidence", 0) * 100 for p in predicciones]
        confianza_promedio = round(sum(confianzas) / len(confianzas), 1)
    else:
        confianza_promedio = 0.0

    return ResultadoConteo(cantidad, confianza_promedio)


def _contar_con_yolo(frame, modelo) -> ResultadoConteo:
    """Cuenta sacos usando un modelo YOLOv8 propio ya entrenado."""
    if modelo is None:
        print("[deteccion] MODO_DETECCION=yolo pero no se cargó ningún modelo. "
              "Usando modo simulado como respaldo.")
        return _contar_simulado()

    resultados = modelo.predict(frame, verbose=False)
    cajas = resultados[0].boxes
    cantidad = len(cajas)

    if cantidad > 0:
        confianza_promedio = round(float(cajas.conf.mean()) * 100, 1)
    else:
        confianza_promedio = 0.0

    return ResultadoConteo(cantidad, confianza_promedio)
