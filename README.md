# Monitor de Seguridad Vial — Interfaz en Streamlit

Interfaz interactiva del proyecto. Usa el mismo pipeline del notebook `monitor_vial.ipynb`:
YOLO11 + ByteTrack + tres ROI, con conteo, permanencia, contravía, velocidad y alertas.

## Archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Interfaz de Streamlit |
| `monitor_vial_core.py` | Pipeline del sistema empaquetado como módulo |
| `requirements.txt` | Dependencias |

## Cómo ejecutarla

1. Instalar las dependencias (una sola vez):

   ```
   pip install -r requirements.txt
   ```

2. El video del proyecto (`video_calle.mp4`) ya está incluido en esta carpeta. También se puede subir otro video desde la interfaz.

3. Iniciar la aplicación:

   ```
   streamlit run app.py
   ```

   Se abre en el navegador, en `http://localhost:8501`.

## Uso

1. En la barra lateral, elegir el video: el del proyecto o uno subido desde la interfaz. Tiene que ser de la misma cámara y con el mismo encuadre.
2. Ajustar los parámetros si hace falta:
   - Modelo YOLO.
   - Confianza mínima.
   - Salto de frames.
   - ROI activas.
   - Permanencia máxima de cada ROI.
   - Límite de velocidad.
3. Para una prueba rápida, poner un valor en **Procesar solo los primeros N segundos**.
4. Pulsar **▶ Procesar video**.
5. Revisar los resultados en la página:
   - Video procesado.
   - Estadísticas finales: total de objetos, visitas por ROI, permanencia media, velocidad media, contravías y alertas.
   - Pestañas de detalle.
6. Descargar el Excel, los CSV o el video procesado.

## Versión web (Streamlit Community Cloud)

La app está desplegada en Streamlit Community Cloud. No requiere instalar nada: el procesamiento corre en la CPU del servidor, así que por defecto se procesan solo los primeros 30 s del video.

Archivos de despliegue:

| Archivo | Para qué sirve |
|---|---|
| `requirements.txt` | Dependencias de Python (PyTorch en versión CPU) |
| `packages.txt` | Librería del sistema `libgl1`, que necesita OpenCV |
| `.streamlit/config.toml` | Tema visual de la interfaz |
| `video_calle.mp4` | Video del proyecto, disponible en la app sin subir nada |

## Tiempos orientativos

En CPU, 1 minuto de video tarda unos 3 minutos con `yolo11s` y 1 de cada 2 frames. Con GPU, unos segundos.
