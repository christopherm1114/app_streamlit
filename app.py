"""
Interfaz interactiva del Monitor de Seguridad Vial (Streamlit).

Flujo de la página:
  1. Barra lateral  → el usuario elige el video y ajusta los parámetros del pipeline.
  2. Vista previa   → primer frame con las ROI dibujadas + ficha técnica del video.
  3. Procesamiento  → SistemaMonitor.procesar() corre YOLO + ByteTrack frame a frame.
  4. Resultados     → KPIs, video anotado, gráficos, tablas de detalle y descargas.

Streamlit re-ejecuta este script completo en cada interacción, por eso los resultados
del procesamiento (costoso) se guardan en st.session_state y no se recalculan.

Ejecutar con:   streamlit run app.py
"""
import os
import tempfile

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from monitor_vial_core import ROIS_BASE, Parametros, SistemaMonitor, convertir_h264

# =====================================================================
# Configuración general de la página
# =====================================================================
# El tema (colores y fuentes) está en .streamlit/config.toml.
st.set_page_config(page_title="Monitor Vial", page_icon=":material/traffic:", layout="wide")

# Colores de las ROI: los mismos que se dibujan sobre el video (en el core están en BGR),
# así el usuario relaciona cada gráfico con la zona que ve en pantalla.
COLOR_ROI = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
# Colores por clase de objeto: coinciden con PALETA_CLASES del core (cajas del video).
COLOR_CLASE = {"auto": "#2a78d6", "moto": "#eb6834", "camion": "#1baf7a", "bus": "#eda100",
               "persona": "#e87ba4", "bicicleta": "#008300"}
COLOR_ALERTA = "#EF4444"
VIDEO_PROYECTO = "video_calle.mp4"

# Ajustes finos de estilo que el tema nativo no cubre. Se usan selectores data-testid
# (estables en Streamlit) y se respeta prefers-reduced-motion.
st.markdown("""
<style>
/* Texto oscuro sobre el botón primario verde: contraste ≥ 4.5:1 (blanco sobre verde no llega) */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] * { color: #0F172A !important; font-weight: 600; }
/* Etiquetas de métricas en versalitas discretas; valores en fuente monoespaciada (cifras alineadas) */
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: .04em; font-size: .75rem; color: #94A3B8; }
[data-testid="stMetricValue"] { font-family: "Fira Code", monospace; font-size: 1.6rem; }
/* "Eyebrow" sobre el título principal */
.mv-eyebrow { color: #4ADE80; font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-bottom: -.6rem; }
/* Menos espacio vacío arriba del contenido */
.block-container { padding-top: 2.2rem; }
/* Transiciones suaves en botones (desactivadas si el usuario pide menos movimiento) */
button { transition: filter .15s ease, transform .15s ease; }
button:hover { filter: brightness(1.08); }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""", unsafe_allow_html=True)

# Carpeta temporal por sesión: aquí van el video subido y los videos procesados.
# Se crea una sola vez por sesión (session_state sobrevive a las re-ejecuciones).
if "carpeta" not in st.session_state:
    st.session_state.carpeta = tempfile.mkdtemp(prefix="monitor_vial_")
carpeta = st.session_state.carpeta

# =====================================================================
# Barra lateral: selección del video y parámetros del pipeline
# =====================================================================
with st.sidebar:
    st.title(":material/traffic: Monitor Vial")
    st.caption("YOLO11 · ByteTrack · Regiones de interés")

    # ---------- 1. Fuente del video ----------
    st.subheader(":material/movie: 1 · Video", divider="gray")
    # Solo se ofrece el video del proyecto si existe junto a app.py.
    opciones = (["Video del proyecto"] if os.path.exists(VIDEO_PROYECTO) else []) + ["Subir un video"]
    fuente = st.segmented_control("Fuente del video", opciones, default=opciones[0],
                                  label_visibility="collapsed", width="stretch") or opciones[0]
    ruta_video = None
    if fuente == "Video del proyecto":
        ruta_video = VIDEO_PROYECTO
    else:
        subido = st.file_uploader("Video de la misma cámara", type=["mp4", "mov", "avi", "mkv"],
                                  help="Las ROI y la calibración de velocidad son de este encuadre concreto.")
        if subido is not None:
            # OpenCV necesita una ruta en disco: se guarda el archivo subido en la carpeta de la sesión
            # (solo la primera vez, para no reescribirlo en cada re-ejecución).
            ruta_video = os.path.join(carpeta, "entrada_" + subido.name)
            if not os.path.exists(ruta_video):
                with open(ruta_video, "wb") as f:
                    f.write(subido.getbuffer())

    # ---------- 2. Detección (YOLO) ----------
    st.subheader(":material/center_focus_strong: 2 · Detección", divider="gray")
    # Modelo fijo: YOLO11s (el del proyecto, incluido en el repo; equilibrio entre precisión y velocidad en CPU).
    modelo = "yolo11s.pt"
    st.caption(":material/memory: Modelo: **YOLO11s**")
    conf = st.slider("Confianza mínima", 0.10, 0.80, 0.25, 0.05,
                     help="Detecciones con menor confianza se descartan. Más alto = menos falsos positivos, más objetos perdidos.")
    salto = st.select_slider("Procesar 1 de cada N frames", options=[1, 2, 3, 4], value=2,
                             help="2 = 15 fps efectivos en un video de 30 fps (configuración del proyecto)")

    # ---------- 3. Zonas (ROI) y umbrales de alerta ----------
    st.subheader(":material/pentagon: 3 · Zonas y alertas", divider="gray")
    rois_activas = st.multiselect("ROI activas", list(ROIS_BASE), default=list(ROIS_BASE),
                                  format_func=lambda k: ROIS_BASE[k]["nombre"])
    # Un tiempo máximo de permanencia por cada ROI activa (dict clave → segundos).
    t_max = {k: st.number_input(f"Permanencia máxima en {k} (s)", 1.0, 120.0, float(ROIS_BASE[k]["t_max"]), 1.0)
             for k in rois_activas}
    limite = st.slider("Límite de velocidad (km/h)", 10, 100, 50, 5, help="Solo carril A (zona calibrada)")

    # ---------- 4. Duración a procesar ----------
    st.subheader(":material/schedule: 4 · Duración", divider="gray")
    # Por defecto 30 s: en la CPU de Streamlit Community Cloud el video completo tardaría varios minutos.
    max_seg = st.number_input("Procesar solo los primeros N segundos", 0, 7200, 30, 10,
                              help="0 = todo el video. En la versión web conviene ≤ 60 s (se procesa en CPU).")

# =====================================================================
# Encabezado del contenido principal
# =====================================================================
st.markdown('<div class="mv-eyebrow">Computer Vision · Seguridad vial</div>', unsafe_allow_html=True)
st.title("Monitor de Seguridad Vial")
st.markdown("Detecta, rastrea y cuenta vehículos y personas en tres regiones de interés; mide su "
            "**tiempo de permanencia**, detecta **contravía**, estima la **velocidad** en el carril cercano "
            "y genera **alertas**.")

# ---------- Estado vacío: aún no hay video o no hay ROI ----------
if ruta_video is None:
    st.info("Sube un video de la cámara en la barra lateral para comenzar.", icon=":material/upload:")
    # Mini guía de uso en tres pasos para orientar al usuario nuevo.
    pasos = [(":material/movie:", "1. Elige el video", "Del mismo encuadre de cámara que el proyecto."),
             (":material/tune:", "2. Ajusta parámetros", "Modelo, confianza, ROI activas y umbrales."),
             (":material/play_circle:", "3. Procesa", "Obtén video anotado, estadísticas y alertas.")]
    for col, (icono, titulo, texto) in zip(st.columns(3), pasos):
        with col.container(border=True):
            st.markdown(f"#### {icono} {titulo}")
            st.caption(texto)
    st.stop()
if not rois_activas:
    st.warning("Selecciona al menos una ROI en la barra lateral.", icon=":material/warning:")
    st.stop()

# =====================================================================
# Preparación del sistema y vista previa
# =====================================================================
# Construir SistemaMonitor es barato (lee metadatos, escala ROI y homografía); el modelo YOLO
# se carga de forma perezosa recién al procesar.
params = Parametros(modelo=modelo, conf=conf, salto_frames=salto, limite_velocidad=float(limite),
                    t_max=t_max, rois_activas=tuple(rois_activas))
sistema = SistemaMonitor(ruta_video, params)
duracion = sistema.total_frames / sistema.fps
a_procesar = duracion if not max_seg else min(duracion, max_seg)

col_prev, col_info = st.columns([3, 2], gap="medium")
with col_prev:
    # Primer frame con las ROI superpuestas: permite verificar que el encuadre coincide.
    primer = sistema.leer_frame(0)
    if primer is not None:
        st.image(cv2.cvtColor(sistema.dibujar_rois(primer), cv2.COLOR_BGR2RGB),   # OpenCV usa BGR; el navegador RGB
                 caption="Primer frame con las ROI y el sentido de circulación permitido", width="stretch")
with col_info:
    with st.container(border=True):
        st.markdown("##### :material/info: Ficha del video")
        st.caption(f"`{os.path.basename(ruta_video)}`")
        # Datos técnicos en una cuadrícula 2×2 de métricas.
        f1, f2 = st.columns(2)
        f1.metric("Resolución", f"{sistema.W}×{sistema.H}", help=f"Escala ×{sistema.escala:.2f} respecto a 704×480")
        f2.metric("FPS procesados", f"{sistema.fps_efectivo:.1f}", help=f"FPS originales: {sistema.fps:.2f}")
        f3, f4 = st.columns(2)
        f3.metric("Duración", f"{duracion:.1f} s", help=f"{sistema.total_frames} frames")
        f4.metric("A procesar", f"{a_procesar:.0f} s")
        # Chips con la configuración elegida, para confirmarla de un vistazo antes de procesar.
        with st.container(horizontal=True, gap="small"):
            st.badge(modelo.removesuffix(".pt"), icon=":material/memory:", color="blue")
            st.badge(f"conf ≥ {conf:.2f}", icon=":material/filter_alt:", color="gray")
            st.badge(f"ROI {' · '.join(rois_activas)}", icon=":material/pentagon:", color="green")
        procesar = st.button("Procesar video", type="primary", icon=":material/play_arrow:", width="stretch")
    st.caption("Las ROI y la calibración de velocidad son de esta cámara y este encuadre. "
               "Funciona con cualquier duración y con 704×480 o 1920×1080.")

# =====================================================================
# Procesamiento (solo cuando se pulsa el botón)
# =====================================================================
if procesar:
    # st.status agrupa el progreso en un bloque plegable con estado (en curso / completo).
    with st.status("Procesando video…", expanded=True) as estado:
        barra = st.progress(0.0, text="Cargando modelo…")
        salida = os.path.join(carpeta, "video_procesado.mp4")
        # El core llama a `progreso(fracción, texto)` cada 10 frames procesados.
        res = sistema.procesar(salida, max_segundos=max_seg or None, progreso=lambda f, t: barra.progress(f, text=t))
        barra.progress(1.0, text="Convirtiendo el video para el navegador…")
        # OpenCV escribe mp4v, que los navegadores no reproducen: se recodifica a H.264.
        st.session_state.video_web = convertir_h264(salida, os.path.join(carpeta, "video_procesado_h264.mp4"))
        # Se guardan los resultados para que sobrevivan a las re-ejecuciones de Streamlit.
        st.session_state.resultados = res
        st.session_state.video_origen = ruta_video
        estado.update(label=f"Procesamiento completo · {res.frames_procesados} frames analizados",
                      state="complete", expanded=False)

# Solo se muestran resultados si existen y corresponden al video seleccionado actualmente.
res = st.session_state.get("resultados")
if res is None or st.session_state.get("video_origen") != ruta_video:
    st.stop()

# =====================================================================
# Resultados: indicadores clave (KPIs)
# =====================================================================
st.divider()
st.header(":material/insights: Resultados")
e = res.estadisticas()
c = st.columns(6)
c[0].metric("Objetos detectados", e["total_objetos"], border=True,
            help="IDs de tracking vistos en al menos k frames (descarta detecciones fugaces)")
c[1].metric("Visitas a las ROI", e["visitas_totales"], border=True, help="Cada reingreso a una ROI cuenta como visita nueva")
c[2].metric("Permanencia media", f"{e['permanencia_media_s']:.1f} s" if e["visitas_totales"] else "—", border=True)
c[3].metric("Velocidad media (A)", f"{e['velocidad_media_kmh']:.1f} km/h" if len(res.velocidades) else "—", border=True)
c[4].metric("Contravías", e["contravias"], border=True)
c[5].metric("Alertas", e["alertas"], border=True, help="Contravía + exceso de velocidad + permanencia excesiva")

# Distribución por clase como chips (más legible que una línea de texto).
if len(res.total_por_clase):
    with st.container(horizontal=True, gap="small"):
        for clase, n in res.total_por_clase.items():
            st.badge(f"{clase}: {n}", color="gray")

# ---------- Video anotado + resumen por ROI ----------
col_v, col_r = st.columns([3, 2], gap="medium")
with col_v:
    st.video(st.session_state.video_web)
with col_r:
    with st.container(border=True):
        st.markdown("##### :material/pentagon: Resumen por ROI")
        st.dataframe(res.resumen, hide_index=True, width="stretch")
        # Se añade la clave (A/B/C) para colorear cada barra con el color de su ROI.
        datos = res.resumen.assign(clave=list(res.sistema.rois))
        st.altair_chart(
            alt.Chart(datos, title="Visitas por ROI").mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("ROI:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Visitas:Q", title="Visitas"),
                color=alt.Color("clave:N", scale=alt.Scale(domain=list(COLOR_ROI), range=list(COLOR_ROI.values())), legend=None),
                tooltip=["ROI", "Visitas", "Objetos únicos", "Permanencia media (s)"]).properties(height=220),
            width="stretch")

# =====================================================================
# Pestañas de detalle
# =====================================================================
n_alertas = len(res.alertas)
t1, t2, t3, t4, t5 = st.tabs([":material/category: Conteo por clase", ":material/timer: Permanencia",
                              ":material/speed: Velocidades", f":material/warning: Alertas ({n_alertas})",
                              ":material/table_rows: Visitas (detalle)"])

# ---------- Conteo de objetos únicos por clase y ROI ----------
with t1:
    if len(res.conteo_clase):
        st.dataframe(res.conteo_clase, width="stretch")
        # Formato "largo" (clase, ROI, objetos) para un gráfico de barras agrupadas con xOffset.
        largo = res.conteo_clase.drop(columns="TOTAL").reset_index().melt("clase", var_name="ROI", value_name="objetos")
        st.altair_chart(alt.Chart(largo).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("clase:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="ROI:N", y=alt.Y("objetos:Q", title="Objetos únicos"),
            color=alt.Color("ROI:N", scale=alt.Scale(domain=list(res.conteo_clase.columns[:-1]),
                                                     range=[COLOR_ROI[k] for k in res.sistema.rois])),
            tooltip=["clase", "ROI", "objetos"]).properties(height=320), width="stretch")
    else:
        st.info("No se registraron visitas.", icon=":material/info:")

# ---------- Distribución de la permanencia por ROI ----------
with t2:
    if len(res.visitas):
        # Boxplot: mediana, cuartiles y valores atípicos de la permanencia en cada ROI.
        st.altair_chart(alt.Chart(res.visitas).mark_boxplot(size=40).encode(
            x=alt.X("roi:N", title="ROI", axis=alt.Axis(labelAngle=0)), y=alt.Y("permanencia_s:Q", title="Permanencia (s)"),
            color=alt.Color("roi:N", scale=alt.Scale(domain=list(COLOR_ROI), range=list(COLOR_ROI.values())), legend=None)
        ).properties(height=320), width="stretch")
        st.dataframe(res.visitas.groupby("roi")["permanencia_s"].describe().round(2), width="stretch")
    else:
        st.info("No se registraron visitas.", icon=":material/info:")

# ---------- Velocidades medidas en el carril A ----------
with t3:
    vel = res.velocidades
    if len(vel):
        a, b, c3 = st.columns(3)
        a.metric("Media", f"{vel.velocidad_kmh.mean():.1f} km/h", border=True)
        b.metric("Percentil 85", f"{vel.velocidad_kmh.quantile(0.85):.1f} km/h", border=True,
                 help="Velocidad que no supera el 85 % de los vehículos (referencia habitual en ingeniería de tránsito)")
        c3.metric(f"Sobre {limite} km/h", f"{(vel.velocidad_kmh > limite).mean() * 100:.0f} %", border=True)
        # Un punto por vehículo + línea roja discontinua en el límite de velocidad.
        puntos = alt.Chart(vel).mark_circle(size=90, opacity=0.85).encode(
            x=alt.X("velocidad_kmh:Q", title="Velocidad (km/h)"), y=alt.Y("clase:N", title=None),
            color=alt.Color("clase:N", scale=alt.Scale(domain=list(COLOR_CLASE), range=list(COLOR_CLASE.values())), legend=None),
            tooltip=["id", "clase", "velocidad_kmh", "t_entrada_s"])
        regla = alt.Chart(pd.DataFrame({"x": [limite]})).mark_rule(color=COLOR_ALERTA, strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((puntos + regla).properties(height=260), width="stretch")
        st.dataframe(vel[["id", "clase", "t_entrada_s", "velocidad_kmh", "exceso_velocidad"]], hide_index=True, width="stretch",
                     column_config={"t_entrada_s": st.column_config.NumberColumn("Entrada (s)", format="%.1f"),
                                    "velocidad_kmh": st.column_config.NumberColumn("Velocidad", format="%.1f km/h"),
                                    "exceso_velocidad": st.column_config.CheckboxColumn("Exceso")})
    else:
        st.info("No hay velocidades medidas (la velocidad solo se mide en la ROI A).", icon=":material/info:")

# ---------- Registro de alertas ----------
with t4:
    if n_alertas:
        st.dataframe(res.alertas, hide_index=True, width="stretch",
                     column_config={"t_s": st.column_config.NumberColumn("Tiempo (s)", format="%.1f"),
                                    "tipo": st.column_config.TextColumn("Tipo"),
                                    "detalle": st.column_config.TextColumn("Detalle", width="large")})
    else:
        st.success("Sin alertas: ningún vehículo circuló en contravía, superó el límite ni permaneció más de lo permitido.",
                   icon=":material/check_circle:")

# ---------- Todas las visitas (una fila por entrada a una ROI) ----------
with t5:
    st.dataframe(res.visitas, hide_index=True, width="stretch")

# =====================================================================
# Exportación de resultados
# =====================================================================
with st.container(border=True):
    st.markdown("##### :material/download: Exportar")
    d = st.columns(5)
    # utf-8-sig añade BOM para que Excel abra bien los acentos de los CSV.
    d[0].download_button("Excel completo", res.a_excel(), "estadisticas_monitor_vial.xlsx",
                         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                         icon=":material/table_view:", type="primary", width="stretch")
    d[1].download_button("visitas.csv", res.visitas.to_csv(index=False).encode("utf-8-sig"), "visitas.csv", "text/csv",
                         icon=":material/description:", width="stretch")
    d[2].download_button("alertas.csv", res.alertas.to_csv(index=False).encode("utf-8-sig"), "alertas.csv", "text/csv",
                         icon=":material/description:", width="stretch")
    d[3].download_button("resumen_roi.csv", res.resumen.to_csv(index=False).encode("utf-8-sig"), "resumen_roi.csv", "text/csv",
                         icon=":material/description:", width="stretch")
    with open(st.session_state.video_web, "rb") as f:
        d[4].download_button("Video procesado", f.read(), "video_procesado.mp4", "video/mp4",
                             icon=":material/movie:", width="stretch")
