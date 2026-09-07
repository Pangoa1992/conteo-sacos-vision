# conteo-sacos-vision

Módulo de visión computarizada del Sistema de Conteo de Sacos — MACROMEC.

## Estado actual

✅ **3 modos de detección disponibles**, elegibles con `MODO_DETECCION` en el `.env`:

| Modo | Cuándo usarlo | Requiere |
|---|---|---|
| `simulado` (por defecto) | Para probar el resto del sistema sin depender de nada | Nada |
| `sam3` | **Recomendado por ahora** — detecta sacos sin entrenar nada, usando el modelo SAM3 vía Roboflow | Cuenta gratis en Roboflow + API key |
| `yolo` | Modelo propio entrenado con fotos reales de la fábrica (para producción final, sin depender de internet) | Modelo `.pt` ya entrenado (ver más abajo) |

**Hallazgo importante:** describir el objeto como `"saco de zanahoria"` da resultados pobres en SAM3. Describirlo por su **apariencia visual**, `"orange mesh bag"`, detectó 8-10 de 8 sacos reales en una prueba — mucho mejor.

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edita el `.env` según el modo que quieras usar:

**Para probar con SAM3 (recomendado):**
```
MODO_DETECCION=sam3
ROBOFLOW_API_KEY=tu_api_key_de_roboflow
ROBOFLOW_WORKFLOW_URL=https://serverless.roboflow.com/willy-urzch/workflows/carrot-sack-counter
```
(La API key se obtiene en tu proyecto de Roboflow, pestaña "Overview" → "Model Endpoint" → "API Key")

Luego corre:
```bash
python -m src.main
```

## Entrenar tu propio modelo YOLOv8 (modo `yolo`)

Existe un pipeline de entrenamiento probado (ver notebook `Conteo_Sacos_Entrenamiento_Validacion.ipynb` compartido con el equipo). Requiere:
1. Fotos reales etiquetadas (bounding boxes) en formato YOLO.
2. Correr el entrenamiento (`ultralytics` YOLO, transfer learning desde `yolov8n.pt`).
3. Apuntar `RUTA_MODELO_YOLO` en el `.env` al archivo `best.pt` resultante.

**Nota:** con menos de ~30-50 fotos de entrenamiento, el modelo no generaliza bien (probado). Se recomienda juntar más fotos siguiendo esta guía antes de entrenar:
- Fotos variadas: sacos separados y sacos pegados entre sí
- Distintos ángulos y momentos del día (luz)
- Distinta cantidad de sacos por foto (pocos y muchos)

## Pendiente

- [ ] Confirmar formato exacto de la respuesta de Roboflow con la API key real (el código soporta las 2 estructuras más probables, falta validar con datos reales)
- [ ] Juntar 30-50+ fotos propias para entrenar el modelo YOLOv8 final
- [ ] Ajustar `iou_threshold` en SAM3 para eliminar detecciones duplicadas de un mismo saco
- [ ] Definir `INTERVALO_FRAMES` según el FPS real de la cámara IP (cuando la faciliten)
- [ ] Agregar token de autenticación (`VISION_API_TOKEN`) validado en el backend


