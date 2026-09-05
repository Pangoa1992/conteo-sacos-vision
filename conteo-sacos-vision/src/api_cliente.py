"""
Cliente que envía los resultados del conteo al backend (módulo web)
mediante una petición POST a /api/conteo.
"""

import os
import requests

from .deteccion import ResultadoConteo


def enviar_conteo(resultado: ResultadoConteo, fuente_camara: str = "camara_poza_1") -> dict:
    url = os.getenv("BACKEND_API_URL", "http://localhost:3000/api/conteo")
    token = os.getenv("VISION_API_TOKEN", "")

    payload = {
        "cantidadSacos": resultado.cantidad_sacos,
        "fuenteCamara": fuente_camara,
        "confianzaPromedio": resultado.confianza_promedio,
    }
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    try:
        respuesta = requests.post(url, json=payload, headers=headers, timeout=10)
        respuesta.raise_for_status()
        return respuesta.json()
    except requests.RequestException as error:
        # No se detiene el pipeline por un fallo de red puntual;
        # se registra el error para revisión posterior.
        print(f"[api_cliente] No se pudo enviar el conteo al backend: {error}")
        return {"error": str(error)}
