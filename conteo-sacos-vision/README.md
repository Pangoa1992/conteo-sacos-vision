# conteo-sacos-vision

Módulo de visión computarizada del Sistema de Conteo de Sacos — MACROMEC.

## Estado actual

⚠️ **Detección simulada (stub).** `src/deteccion.py` genera un conteo aleatorio
realista (90-130 sacos) en vez de usar un modelo real, porque todavía no se
cuenta con:
- Video/fotos reales de la fábrica para entrenar o adaptar el modelo.
- Definición de si se usará un modelo YOLOv8 preentrenado adaptado o entrenado desde cero.

Esto permite probar **todo el pipeline end-to-end** (captura → detección → envío al backend)
desde ya, y dejar solo la función `contar_sacos()` pendiente de reemplazar cuando
haya datos reales.

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # y completar BACKEND_API_URL / VIDEO_SOURCE
python -m src.main
```

Para probar sin cámara IP: coloca un video `.mp4` en `videos_prueba/` y apunta
`VIDEO_SOURCE` a esa ruta en el `.env`.

## Pendiente

- [ ] Conseguir video/fotos reales de la fábrica (poza de selección/empaquetado)
- [ ] Decidir: modelo YOLOv8 preentrenado adaptado vs. entrenado desde cero
- [ ] Reemplazar `contar_sacos()` en `src/deteccion.py` con la inferencia real
- [ ] Definir `INTERVALO_FRAMES` según el FPS real de la cámara IP
- [ ] Agregar token de autenticación (`VISION_API_TOKEN`) validado en el backend
