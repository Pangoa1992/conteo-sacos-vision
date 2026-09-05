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

✅ **Ya soporta 3 formas de obtener imágenes**, para poder empezar a probar sin
esperar a tener la cámara IP de la fábrica:

| Fuente | Cómo configurarla en `.env` | Cuándo usarla |
|---|---|---|
| 📷 Fotos sueltas | `VIDEO_SOURCE=./fotos_prueba` | Tomas con el celular en la fábrica, mientras no hay cámara IP |
| 🎞️ Video grabado | `VIDEO_SOURCE=./videos_prueba/muestra.mp4` | Cuando ya se grabó un video real de la poza |
| 📡 Cámara en vivo | `VIDEO_SOURCE=rtsp://192.168.1.50/stream` | Cuando MACROMEC facilite acceso a la cámara IP de vigilancia |

El programa detecta automáticamente cuál de los tres estás usando — no hay que
cambiar nada más en el código.

## Cómo correrlo

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # y completar VIDEO_SOURCE con una de las 3 opciones de arriba
python -m src.main
```

Para probar con fotos: coloca tus imágenes en `fotos_prueba/` (o la carpeta que
prefieras) y apunta `VIDEO_SOURCE` ahí. Para probar con video: coloca el archivo
en `videos_prueba/`.

## Pendiente

- [ ] Conseguir fotos/video reales de la fábrica (poza de selección/empaquetado)
- [ ] Decidir: modelo YOLOv8 preentrenado adaptado vs. entrenado desde cero
- [ ] Reemplazar `contar_sacos()` en `src/deteccion.py` con la inferencia real
- [ ] Definir `INTERVALO_FRAMES` según el FPS real de la cámara IP (cuando la faciliten)
- [ ] Agregar token de autenticación (`VISION_API_TOKEN`) validado en el backend

