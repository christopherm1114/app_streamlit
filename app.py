"""
Interfaz interactiva del Monitor de Seguridad Vial (Streamlit).

Ejecutar con:   streamlit run app.py
"""
import os
import tempfile

import altair as alt
import cv2
import pandas as pd
import streamlit as st

from monitor_vial_core import CLASES, ROIS_BASE, Parametros, SistemaMonitor, convertir_h264

st.set_page_config(page_title="Monitor Vial", page_icon="🚦", layout="wide")

COLOR_ROI = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}          # Mismos colores que en el video
COLOR_CLASE = {"auto": "#2a78d6", "moto": "#eb6834", "camion": "#1baf7a", "bus": "#eda100",
               "persona": "#e87ba4", "bicicleta": "#008300"}
VIDEO_PROYECTO = "video_calle.mp4"

if "carpeta" not in st.session_state:
    st.session_state.carpeta = tempfile.mkdtemp(prefix="monitor_vial_")
carpeta = st.session_state.carpeta

# =====================================================================
# Barra lateral: video y parámetros
# =====================================================================
st.sidebar.title("🚦 Monitor Vial")
st.sidebar.caption("YOLO11 + ByteTrack + Regiones de Interés")

st.sidebar.header("1. Video")
opciones = (["Video del proyecto"] if os.path.exists(VIDEO_PROYECTO) else []) + ["Subir un video"]
fuente = st.sidebar.radio("Fuente", opciones, label_visibility="collapsed")
ruta_video = None
if fuente == "Video del proyecto":
    ruta_video = VIDEO_PROYECTO
else:
    subido = st.sidebar.file_uploader("Video de la misma cámara (mp4, mov, avi)", type=["mp4", "mov", "avi", "mkv"])
    if subido is not None:
        ruta_video = os.path.join(carpeta, "entrada_" + subido.name)
        if not os.path.exists(ruta_video):
            with open(ruta_video, "wb") as f:
                f.write(subido.getbuffer())

st.sidebar.header("2. Detección")
modelo = st.sidebar.selectbox("Modelo YOLO", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=1,
                              help="n: más rápido · s: equilibrio (usado en el proyecto) · m: más preciso, requiere GPU")
conf = st.sidebar.slider("Confianza mínima", 0.10, 0.80, 0.25, 0.05)
salto = st.sidebar.select_slider("Procesar 1 de cada N frames", options=[1, 2, 3, 4], value=2,
                                 help="2 = 15 fps efectivos en un video de 30 fps (configuración del proyecto)")

st.sidebar.header("3. Zonas y alertas")
rois_activas = st.sidebar.multiselect("ROI activas", list(ROIS_BASE), default=list(ROIS_BASE),
                                      format_func=lambda k: ROIS_BASE[k]["nombre"])
t_max = {k: st.sidebar.number_input(f"Permanencia máxima en {k} (s)", 1.0, 120.0, float(ROIS_BASE[k]["t_max"]), 1.0)
         for k in rois_activas}
limite = st.sidebar.slider("Límite de velocidad (km/h)", 10, 100, 50, 5, help="Solo carril A (zona calibrada)")

st.sidebar.header("4. Duración")
max_seg = st.sidebar.number_input("Procesar solo los primeros N segundos (0 = todo el video)", 0, 7200, 0, 10)

# =====================================================================
# Contenido principal
# =====================================================================
st.title("Monitor de Seguridad Vial")
st.markdown("Detecta, rastrea y cuenta vehículos y personas en tres regiones de interés; mide su **tiempo de permanencia**, "
            "detecta **contravía**, estima la **velocidad** en el carril cercano y genera **alertas**.")

if ruta_video is None:
    st.info("Sube un video de la cámara en la barra lateral para comenzar.")
    st.stop()
if not rois_activas:
    st.warning("Selecciona al menos una ROI.")
    st.stop()

params = Parametros(modelo=modelo, conf=conf, salto_frames=salto, limite_velocidad=float(limite),
                    t_max=t_max, rois_activas=tuple(rois_activas))
sistema = SistemaMonitor(ruta_video, params)
duracion = sistema.total_frames / sistema.fps

col_prev, col_info = st.columns([3, 2])
with col_prev:
    primer = sistema.leer_frame(0)
    if primer is not None:
        st.image(cv2.cvtColor(sistema.dibujar_rois(primer), cv2.COLOR_BGR2RGB),
                 caption="Primer frame con las ROI y el sentido de circulación permitido", use_container_width=True)
with col_info:
    st.subheader("Video")
    st.markdown(f"- **Archivo:** `{os.path.basename(ruta_video)}`\n"
                f"- **Resolución:** {sistema.W} × {sistema.H} px (escala ×{sistema.escala:.2f} respecto a 704×480)\n"
                f"- **FPS:** {sistema.fps:.2f} → {sistema.fps_efectivo:.1f} procesados\n"
                f"- **Duración:** {duracion:.1f} s ({sistema.total_frames} frames)")
    a_procesar = duracion if not max_seg else min(duracion, max_seg)
    st.markdown(f"Se procesarán **{a_procesar:.0f} s** de video.")
    st.caption("Las ROI y la calibración de velocidad son de esta cámara y este encuadre. "
               "Funciona con cualquier duración y con 704×480 o 1920×1080.")
    procesar = st.button("▶ Procesar video", type="primary", use_container_width=True)

if procesar:
    barra = st.progress(0.0, text="Cargando modelo…")
    salida = os.path.join(carpeta, "video_procesado.mp4")
    res = sistema.procesar(salida, max_segundos=max_seg or None, progreso=lambda f, t: barra.progress(f, text=t))
    barra.progress(1.0, text="Convirtiendo el video para el navegador…")
    st.session_state.video_web = convertir_h264(salida, os.path.join(carpeta, "video_procesado_h264.mp4"))
    st.session_state.resultados = res
    st.session_state.video_origen = ruta_video
    barra.empty()

res = st.session_state.get("resultados")
if res is None or st.session_state.get("video_origen") != ruta_video:
    st.stop()

# ---------- Estadísticas finales ----------
st.divider()
st.header("Resultados")
e = res.estadisticas()
c = st.columns(6)
c[0].metric("Objetos detectados", e["total_objetos"])
c[1].metric("Visitas a las ROI", e["visitas_totales"])
c[2].metric("Permanencia media", f"{e['permanencia_media_s']:.1f} s" if e["visitas_totales"] else "—")
c[3].metric("Velocidad media (A)", f"{e['velocidad_media_kmh']:.1f} km/h" if len(res.velocidades) else "—")
c[4].metric("Contravías", e["contravias"])
c[5].metric("Alertas", e["alertas"])
if len(res.total_por_clase):
    st.caption("Objetos por clase: " + " · ".join(f"{k}: {v}" for k, v in res.total_por_clase.items()))

col_v, col_r = st.columns([3, 2])
with col_v:
    st.video(st.session_state.video_web)
with col_r:
    st.subheader("Resumen por ROI")
    st.dataframe(res.resumen, hide_index=True, use_container_width=True)
    datos = res.resumen.assign(clave=[k for k in res.sistema.rois])
    st.altair_chart(
        alt.Chart(datos, title="Visitas por ROI").mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("ROI:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("Visitas:Q", title="Visitas"),
            color=alt.Color("clave:N", scale=alt.Scale(domain=list(COLOR_ROI), range=list(COLOR_ROI.values())), legend=None),
            tooltip=["ROI", "Visitas", "Objetos únicos", "Permanencia media (s)"]).properties(height=240),
        use_container_width=True)

t1, t2, t3, t4, t5 = st.tabs(["Conteo por clase", "Permanencia", "Velocidades", "Alertas", "Visitas (detalle)"])
with t1:
    if len(res.conteo_clase):
        st.dataframe(res.conteo_clase, use_container_width=True)
        largo = res.conteo_clase.drop(columns="TOTAL").reset_index().melt("clase", var_name="ROI", value_name="objetos")
        st.altair_chart(alt.Chart(largo).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("clase:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="ROI:N", y=alt.Y("objetos:Q", title="Objetos únicos"),
            color=alt.Color("ROI:N", scale=alt.Scale(domain=list(res.conteo_clase.columns[:-1]),
                                                     range=[COLOR_ROI[k] for k in res.sistema.rois])),
            tooltip=["clase", "ROI", "objetos"]).properties(height=320), use_container_width=True)
    else:
        st.info("No se registraron visitas.")
with t2:
    if len(res.visitas):
        st.altair_chart(alt.Chart(res.visitas).mark_boxplot(size=40).encode(
            x=alt.X("roi:N", title="ROI", axis=alt.Axis(labelAngle=0)), y=alt.Y("permanencia_s:Q", title="Permanencia (s)"),
            color=alt.Color("roi:N", scale=alt.Scale(domain=list(COLOR_ROI), range=list(COLOR_ROI.values())), legend=None)
        ).properties(height=320), use_container_width=True)
        st.dataframe(res.visitas.groupby("roi")["permanencia_s"].describe().round(2), use_container_width=True)
    else:
        st.info("No se registraron visitas.")
with t3:
    vel = res.velocidades
    if len(vel):
        a, b, c3 = st.columns(3)
        a.metric("Media", f"{vel.velocidad_kmh.mean():.1f} km/h")
        b.metric("Percentil 85", f"{vel.velocidad_kmh.quantile(0.85):.1f} km/h")
        c3.metric(f"Sobre {limite} km/h", f"{(vel.velocidad_kmh > limite).mean() * 100:.0f} %")
        puntos = alt.Chart(vel).mark_circle(size=90, opacity=0.85).encode(
            x=alt.X("velocidad_kmh:Q", title="Velocidad (km/h)"), y=alt.Y("clase:N", title=None),
            color=alt.Color("clase:N", scale=alt.Scale(domain=list(COLOR_CLASE), range=list(COLOR_CLASE.values())), legend=None),
            tooltip=["id", "clase", "velocidad_kmh", "t_entrada_s"])
        regla = alt.Chart(pd.DataFrame({"x": [limite]})).mark_rule(color="#e34948", strokeDash=[5, 4]).encode(x="x:Q")
        st.altair_chart((puntos + regla).properties(height=260), use_container_width=True)
        st.dataframe(vel[["id", "clase", "t_entrada_s", "velocidad_kmh", "exceso_velocidad"]], hide_index=True, use_container_width=True)
    else:
        st.info("No hay velocidades medidas (la velocidad solo se mide en la ROI A).")
with t4:
    if len(res.alertas):
        st.dataframe(res.alertas, hide_index=True, use_container_width=True)
    else:
        st.success("Sin alertas: ningún vehículo circuló en contravía, superó el límite ni permaneció más de lo permitido.")
with t5:
    st.dataframe(res.visitas, hide_index=True, use_container_width=True)

# ---------- Descargas ----------
st.subheader("Exportar")
d = st.columns(5)
d[0].download_button("📊 Excel completo", res.a_excel(), "estadisticas_monitor_vial.xlsx",
                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d[1].download_button("visitas.csv", res.visitas.to_csv(index=False).encode("utf-8-sig"), "visitas.csv", "text/csv", use_container_width=True)
d[2].download_button("alertas.csv", res.alertas.to_csv(index=False).encode("utf-8-sig"), "alertas.csv", "text/csv", use_container_width=True)
d[3].download_button("resumen_roi.csv", res.resumen.to_csv(index=False).encode("utf-8-sig"), "resumen_roi.csv", "text/csv", use_container_width=True)
with open(st.session_state.video_web, "rb") as f:
    d[4].download_button("🎬 Video procesado", f.read(), "video_procesado.mp4", "video/mp4", use_container_width=True)
