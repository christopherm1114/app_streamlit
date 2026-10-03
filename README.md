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

2. Copiar `video_calle.mp4` en esta misma carpeta. Es opcional: también se puede subir un video desde la interfaz.

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

## Tiempos orientativos

En CPU, 1 minuto de video tarda unos 3 minutos con `yolo11s` y 1 de cada 2 frames. Con GPU, unos segundos.
