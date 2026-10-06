"""
monitor_vial_core.py
====================
Pipeline del Monitor de Seguridad Vial (YOLO + ByteTrack + ROI), empaquetado como módulo
para reutilizarlo desde la interfaz de Streamlit (app.py).

Es la misma lógica del notebook monitor_vial.ipynb:
  - Detección con YOLO11 y filtro de conductores de moto/bicicleta.
  - Tracking con ByteTrack (Supervision).
  - Tres ROI (carril cercano, carril lejano, intersección) escaladas a la resolución del video.
  - Conteo sin doble conteo, permanencia, contravía, velocidad (homografía) y alertas.
  - Video anotado y tablas de resultados (pandas) listas para exportar a CSV / Excel.
"""
import copy
import io
import os
import shutil
import subprocess
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=FutureWarning)   # Aviso de deprecación de sv.ByteTrack

# =====================================================================
# Constantes de la escena (cámara de la avenida con parterre)
# =====================================================================
# Subconjunto de clases COCO que interesan (id COCO → nombre). YOLO solo devuelve estas.
CLASES = {0: "persona", 1: "bicicleta", 2: "auto", 3: "moto", 5: "bus", 7: "camion"}
VEHICULOS = {1, 2, 3, 5, 7}
COLOR_ALERTA = (72, 73, 227)                      # Rojo (BGR) reservado para las alertas
RES_BASE = (704, 480)                             # Resolución en la que se dibujaron las ROI

# ROI en la resolución base 704x480 (se escalan a la resolución real del video).
#   nombre    → etiqueta legible.
#   sentido   → vector (dx, dy) en píxeles del sentido de circulación permitido; None = sin control de contravía.
#   t_max     → segundos máximos de permanencia antes de generar alerta.
#   velocidad → True si la ROI está dentro de la zona calibrada para medir velocidad.
#   color     → color BGR con el que se dibuja la ROI.
#   poligono  → vértices (x, y) del polígono de la zona.
ROIS_BASE = {
    "A": {"nombre": "A carril cercano", "sentido": (0.918, 0.397), "t_max": 10.0, "velocidad": True,
          "color": (214, 120, 42),
          "poligono": [(175, 232), (335, 224), (640, 320), (640, 462), (465, 462), (225, 282)]},
    "B": {"nombre": "B carril lejano", "sentido": (-0.965, -0.262), "t_max": 10.0, "velocidad": False,
          "color": (52, 104, 235),
          "poligono": [(375, 200), (650, 274), (650, 304), (375, 229)]},
    "C": {"nombre": "C interseccion", "sentido": None, "t_max": 6.0, "velocidad": False,
          "color": (122, 175, 27),
          "poligono": [(180, 198), (395, 201), (335, 222), (175, 230)]},
}

# Calibración de velocidad: base de los 4 garrafones en 1920x1080 y su posición real (m).
# Con estos 4 pares de puntos se calcula una homografía que lleva píxeles del suelo a metros
# sobre el plano de la calzada (vista "desde arriba"), donde ya se pueden medir distancias reales.
GARRAFONES_1080 = np.float32([[471, 496], [910, 512], [1262, 1057], [1742, 709]])   # G1, G2, G3, G4
GARRAFONES_MUNDO = np.float32([[0, 0], [7.75, 0], [0, 14.75], [7.75, 14.85]])

# Paleta indexada por id de clase COCO (posición 0 = persona, 1 = bicicleta, 2 = auto, ...).
# Los índices 4 y 6 no se usan (gris de relleno).
PALETA_CLASES = sv.ColorPalette.from_hex(
    ["#e87ba4", "#008300", "#2a78d6", "#eb6834", "#888888", "#eda100", "#888888", "#1baf7a"])


# =====================================================================
# Parámetros configurables (la interfaz de Streamlit los modifica)
# =====================================================================
@dataclass
class Parametros:
    # --- Detección ---
    modelo: str = "yolo11s.pt"          # Pesos de YOLO11s (incluidos en el repo)
    conf: float = 0.25                  # Confianza mínima de una detección
    imgsz: int = 704                    # Tamaño de entrada de la red
    salto_frames: int = 2               # Procesar 1 de cada N frames
    # --- Conteo y visitas ---
    k_persistencia: int = 3             # Frames seguidos dentro de la ROI para contar
    tolerancia_salida: int = 8          # Frames fuera de la ROI para cerrar la visita
    # --- Dirección / contravía ---
    ventana: int = 8                    # Posiciones del historial (dirección y velocidad)
    min_desplazamiento_base: float = 12 # px a 704x480 para evaluar la dirección
    umbral_cos_contra: float = -0.5     # cos < -0.5 → se mueve a más de 120° del sentido permitido
    k_contravia: int = 4                # Evaluaciones seguidas en contra para confirmar la contravía
    # --- Filtro de conductores ---
    umbral_conductor: float = 0.30      # Fracción de la caja de la persona cubierta por la moto/bici
    # --- Velocidad ---
    limite_velocidad: float = 50.0      # km/h
    zona_calibrada_y: tuple = (-3.0, 15.0)  # Rango (m) del plano calibrado donde la homografía es fiable
    zona_calibrada_x: tuple = (-1.0, 8.8)
    t_max: dict = field(default_factory=lambda: {"A": 10.0, "B": 10.0, "C": 6.0})
    rois_activas: tuple = ("A", "B", "C")


# =====================================================================
# Lógica de una ROI
# =====================================================================
class MonitorROI:
    """Estado de UNA región de interés: conteo, visitas, permanencia, contravía, velocidad y alertas."""

    def __init__(self, clave, cfg, sistema):
        self.s = sistema                                   # Acceso a FPS, parámetros y homografía
        self.clave, self.nombre, self.color = clave, cfg["nombre"], cfg["color"]
        self.t_max = sistema.p.t_max.get(clave, cfg["t_max"])
        self.poligono = cfg["poligono"]
        # Un objeto está "dentro" si el punto inferior central de su caja (donde toca el suelo)
        # cae en el polígono; así la perspectiva no hace que la parte alta de un camión "invada" otra ROI.
        self.zona = sv.PolygonZone(polygon=cfg["poligono"].astype(np.int64),
                                   triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        self.sentido = None if cfg["sentido"] is None else np.array(cfg["sentido"]) / np.linalg.norm(cfg["sentido"])
        self.mide_velocidad = cfg.get("velocidad", False)

        # --- Estado por ID de tracking ---
        self.racha = defaultdict(int)                      # Frames procesados seguidos dentro de la ROI
        self.inicio_racha, self.ultimo_dentro = {}, {}     # Frame donde empezó la racha / último frame visto dentro
        self.contados, self.conteo_clase = set(), Counter()  # IDs ya contados (sin doble conteo) y conteo por clase
        self.visitas_activas, self.visitas = {}, []        # Visitas en curso (por ID) y visitas cerradas
        self.contra_racha = defaultdict(int)               # Evaluaciones seguidas en sentido contrario
        # IDs que ya dispararon cada tipo de alerta (cada alerta se emite una sola vez por ID)
        self.en_contravia, self.alerta_permanencia, self.alerta_velocidad = set(), set(), set()
        self.velocidad_actual = {}                         # Velocidad mediana actual por ID (para la etiqueta)
        self.ocupacion = 0                                 # Objetos dentro de la ROI en el frame actual

    def actualizar(self, det, idx, clase_de, historial, alertas):
        """Actualiza el estado de la ROI con las detecciones `det` del frame `idx`.

        `historial[tid]` es una deque de (frame, x, y) con las últimas posiciones del ID;
        las alertas nuevas se añaden a la lista compartida `alertas`.
        """
        p, fps = self.s.p, self.s.fps
        t = idx / fps
        dentro = self.zona.trigger(det) if len(det) else np.array([], dtype=bool)
        anclas = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(det) else []
        ids_dentro = set()
        # ---- Recorre solo los objetos que están dentro del polígono ----
        for k in np.where(dentro)[0]:
            tid = int(det.tracker_id[k])
            clase = clase_de(tid)
            ids_dentro.add(tid)
            if self.racha[tid] == 0:
                self.inicio_racha[tid] = idx
            self.racha[tid] += 1
            self.ultimo_dentro[tid] = idx

            # Conteo con persistencia (una vez por ID y ROI; cada reingreso es una visita nueva)
            if self.racha[tid] >= p.k_persistencia and tid not in self.visitas_activas:
                if tid not in self.contados:
                    self.contados.add(tid)
                    self.conteo_clase[clase] += 1
                self.visitas_activas[tid] = {"id": tid, "roi": self.clave, "frame_entrada": self.inicio_racha[tid],
                                             "contravia": False, "alerta_permanencia": False,
                                             "pos_entrada": tuple(anclas[k]), "velocidades": []}
            if tid not in self.visitas_activas:
                continue
            visita = self.visitas_activas[tid]
            visita["pos_salida"] = tuple(anclas[k])

            # Permanencia excesiva (solo vehículos)
            perm = (idx - visita["frame_entrada"]) / fps
            if perm > self.t_max and clase != "persona" and tid not in self.alerta_permanencia:
                self.alerta_permanencia.add(tid)
                visita["alerta_permanencia"] = True
                alertas.append({"t_s": round(t, 2), "frame": idx, "tipo": "permanencia", "roi": self.clave,
                                "id": tid, "clase": clase, "detalle": f"{perm:.0f} s en {self.nombre}"})

            h = historial[tid]
            # Velocidad en la zona calibrada: distancia en metros (vía homografía) entre la posición
            # más antigua y la más reciente de la ventana, dividida por el tiempo transcurrido.
            if self.mide_velocidad and clase != "persona" and len(h) >= p.ventana:
                v = self.s.velocidad_kmh(h[0][1:], h[-1][1:], (h[-1][0] - h[0][0]) / fps)
                if v is not None:
                    visita["velocidades"].append(v)
                    # La mediana de las muestras amortigua el ruido del tracking (saltos de la caja).
                    if len(visita["velocidades"]) >= 3:
                        self.velocidad_actual[tid] = float(np.median(visita["velocidades"]))
                    mediana = float(np.median(visita["velocidades"]))
                    # Se exigen ≥5 muestras antes de alertar, para no disparar con una medición aislada.
                    if len(visita["velocidades"]) >= 5 and mediana > p.limite_velocidad and tid not in self.alerta_velocidad:
                        self.alerta_velocidad.add(tid)
                        alertas.append({"t_s": round(t, 2), "frame": idx, "tipo": "exceso_velocidad", "roi": self.clave,
                                        "id": tid, "clase": clase, "detalle": f"{clase} a {mediana:.0f} km/h"})

            # Contravía: coseno entre el desplazamiento reciente y el sentido permitido
            #   cos ≈ +1 → va a favor;  cos ≈ -1 → va exactamente en contra.
            if self.sentido is not None and clase != "persona" and tid not in self.en_contravia and len(h) >= p.ventana // 2:
                d = np.array(h[-1][1:]) - np.array(h[0][1:])
                dist = np.linalg.norm(d)
                # Objetos casi quietos no se evalúan: su "dirección" sería solo ruido de la caja.
                if dist >= self.s.min_desplazamiento:
                    cos = float(d @ self.sentido) / dist
                    # Histéresis: suma si va en contra, reinicia solo si va claramente a favor
                    # (los movimientos laterales, 0 ≥ cos ≥ umbral, no cambian la racha).
                    if cos < p.umbral_cos_contra:
                        self.contra_racha[tid] += 1
                    elif cos > 0:
                        self.contra_racha[tid] = 0
                    if self.contra_racha[tid] >= p.k_contravia:
                        self.en_contravia.add(tid)
                        visita["contravia"] = True
                        alertas.append({"t_s": round(t, 2), "frame": idx, "tipo": "contravia", "roi": self.clave,
                                        "id": tid, "clase": clase, "detalle": f"{clase} en contravia en {self.nombre}"})

        # ---- Objetos que no están dentro en este frame ----
        # Su racha de entrada se reinicia (deben volver a cumplir k_persistencia para contar).
        for tid in list(self.racha):
            if tid not in ids_dentro:
                self.racha[tid] = 0
        # Una visita se cierra solo tras `tolerancia_salida` frames procesados sin verse dentro:
        # así una oclusión breve o un fallo puntual del detector no parte la visita en dos.
        for tid in list(self.visitas_activas):
            if idx - self.ultimo_dentro[tid] > p.tolerancia_salida * p.salto_frames:
                self._cerrar(tid, clase_de)
        self.ocupacion = len(ids_dentro & set(self.visitas_activas))

    def permanencia_actual(self, tid, idx):
        """Segundos que lleva el ID dentro de la ROI, o None si no tiene una visita activa."""
        v = self.visitas_activas.get(tid)
        return None if v is None else (idx - v["frame_entrada"]) / self.s.fps

    def _cerrar(self, tid, clase_de):
        """Pasa una visita activa a la lista de visitas cerradas con sus métricas finales."""
        fps = self.s.fps
        v = self.visitas_activas.pop(tid)
        v["clase"] = clase_de(tid)
        v["frame_salida"] = self.ultimo_dentro[tid]
        v["t_entrada_s"] = round(v["frame_entrada"] / fps, 2)
        v["t_salida_s"] = round(v["frame_salida"] / fps, 2)
        v["permanencia_s"] = round((v["frame_salida"] - v["frame_entrada"]) / fps, 2)
        v["recorrido_px"] = round(float(np.hypot(*np.subtract(v.pop("pos_salida"), v.pop("pos_entrada")))), 1)
        vels = v.pop("velocidades")
        v["velocidad_kmh"] = round(float(np.median(vels)), 1) if len(vels) >= 3 else np.nan
        v["exceso_velocidad"] = tid in self.alerta_velocidad
        self.visitas.append(v)

    def cerrar_todo(self, clase_de):
        """Cierra las visitas que seguían abiertas al terminar el video."""
        for tid in list(self.visitas_activas):
            self._cerrar(tid, clase_de)


# =====================================================================
# Sistema completo para un video
# =====================================================================
COLUMNAS_VISITAS = ["roi", "id", "clase", "t_entrada_s", "t_salida_s", "permanencia_s", "recorrido_px",
                    "velocidad_kmh", "contravia", "alerta_permanencia", "exceso_velocidad", "frame_entrada", "frame_salida"]


class SistemaMonitor:
    """Prepara ROI, homografía y anotadores según la resolución del video y ejecuta el pipeline."""

    _modelos = {}                                          # Caché de modelos YOLO ya cargados

    def __init__(self, ruta_video, parametros=None):
        self.ruta = ruta_video
        self.p = parametros or Parametros()
        info = sv.VideoInfo.from_video_path(ruta_video)
        self.W, self.H, self.fps, self.total_frames = info.width, info.height, info.fps, info.total_frames
        self.fps_efectivo = self.fps / self.p.salto_frames
        # Factores de escala respecto a 704x480: todo lo definido en píxeles base (ROI, grosores,
        # textos, umbrales de desplazamiento) se adapta a la resolución real del video.
        ex, ey = self.W / RES_BASE[0], self.H / RES_BASE[1]
        self.escala = (ex + ey) / 2
        self.min_desplazamiento = self.p.min_desplazamiento_base * self.escala

        # ROI escaladas (solo las activas)
        self.rois = {}
        for k, r in ROIS_BASE.items():
            if k not in self.p.rois_activas:
                continue
            r = copy.deepcopy(r)
            r["poligono"] = (np.array(r["poligono"]) * [ex, ey]).round().astype(np.int32)
            if r["sentido"] is not None:
                # Si el aspecto cambia (ex ≠ ey) el vector de sentido también se deforma: se escala y renormaliza.
                v = np.array(r["sentido"]) * [ex, ey]
                r["sentido"] = tuple(float(x) for x in v / np.linalg.norm(v))
            self.rois[k] = r

        # Homografía imagen → metros escalada a la resolución del video
        pts = GARRAFONES_1080 * np.float32([self.W / 1920, self.H / 1080])
        self.M = cv2.getPerspectiveTransform(pts, GARRAFONES_MUNDO)

        # Anotadores de Supervision: cajas, etiquetas y trazas coloreadas por clase.
        e = self.escala
        self.anot_cajas = sv.BoxAnnotator(color=PALETA_CLASES, color_lookup=sv.ColorLookup.CLASS, thickness=max(2, round(2 * e)))
        self.anot_etiquetas = sv.LabelAnnotator(color=PALETA_CLASES, color_lookup=sv.ColorLookup.CLASS, text_scale=0.4 * e,
                                                text_padding=round(2 * e), text_color=sv.Color.WHITE)
        self.anot_trazas = sv.TraceAnnotator(color=PALETA_CLASES, color_lookup=sv.ColorLookup.CLASS,
                                             thickness=max(1, round(e)), trace_length=25, position=sv.Position.BOTTOM_CENTER)

    # ---------- utilidades ----------
    @property
    def modelo(self):
        """Modelo YOLO cargado de forma perezosa y cacheado a nivel de clase (se reutiliza entre ejecuciones)."""
        if self.p.modelo not in SistemaMonitor._modelos:
            SistemaMonitor._modelos[self.p.modelo] = YOLO(self.p.modelo)
        return SistemaMonitor._modelos[self.p.modelo]

    def velocidad_kmh(self, p0, p1, dt):
        """Velocidad (km/h) entre dos puntos de imagen separados `dt` segundos, o None si no es medible."""
        # Píxeles → metros en el plano de la calzada.
        (X0, Y0), (X1, Y1) = cv2.perspectiveTransform(np.float32([[p0], [p1]]), self.M).reshape(2, 2)
        zx, zy = self.p.zona_calibrada_x, self.p.zona_calibrada_y
        # Fuera de la zona entre los garrafones la homografía extrapola y el error crece: se descarta.
        en_zona = lambda X, Y: zx[0] <= X <= zx[1] and zy[0] <= Y <= zy[1]
        if dt <= 0 or not (en_zona(X0, Y0) and en_zona(X1, Y1)):
            return None
        return float(np.hypot(X1 - X0, Y1 - Y0) / dt * 3.6)   # m/s → km/h

    def filtrar_conductores(self, det):
        """Elimina las 'personas' que en realidad son conductores de moto o bicicleta.

        YOLO detecta por separado a la persona y al vehículo de dos ruedas; sin este filtro el
        conductor se contaría como peatón. Se descarta la persona cuya caja queda cubierta en más
        de `umbral_conductor` por la caja de una moto/bici (intersección / área de la persona).
        """
        if len(det) == 0:
            return det
        es_persona, es_rueda = det.class_id == 0, np.isin(det.class_id, [1, 3])
        if not es_persona.any() or not es_rueda.any():
            return det
        P, R = det.xyxy[es_persona], det.xyxy[es_rueda]
        # Intersección vectorizada de todas las parejas persona × rueda (broadcasting P[:, None] vs R[None, :]).
        ix = np.clip(np.minimum(P[:, None, 2], R[None, :, 2]) - np.maximum(P[:, None, 0], R[None, :, 0]), 0, None)
        iy = np.clip(np.minimum(P[:, None, 3], R[None, :, 3]) - np.maximum(P[:, None, 1], R[None, :, 1]), 0, None)
        area = (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1])
        conductor = (ix * iy / np.maximum(area[:, None], 1e-6) > self.p.umbral_conductor).any(axis=1)
        mantener = np.ones(len(det), dtype=bool)
        mantener[np.where(es_persona)[0][conductor]] = False
        return det[mantener]

    def leer_frame(self, n=0):
        """Lee un único frame (por defecto el primero) para la vista previa."""
        cap = cv2.VideoCapture(self.ruta)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    # ---------- dibujo ----------
    def dibujar_rois(self, img, alpha=0.25):
        """Dibuja las ROI semitransparentes, su letra y una flecha con el sentido permitido."""
        e = self.escala
        capa = img.copy()
        for r in self.rois.values():
            cv2.fillPoly(capa, [r["poligono"]], r["color"])
        img = cv2.addWeighted(capa, alpha, img, 1 - alpha, 0)
        gr = max(2, round(2 * e))
        for k, r in self.rois.items():
            cv2.polylines(img, [r["poligono"]], True, r["color"], gr)
            cx, cy = r["poligono"].mean(axis=0).astype(int)
            cv2.putText(img, k, (cx - int(8 * e), cy + int(6 * e)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * e, (255, 255, 255), gr)
            if r["sentido"] is not None:
                p1 = np.array([cx + 18 * e, cy])
                p2 = p1 + 55 * e * np.array(r["sentido"])
                cv2.arrowedLine(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (255, 255, 255), gr, tipLength=0.3)
        return img

    def _panel(self, img, monitores, alertas, t):
        """Panel semitransparente arriba a la izquierda: tiempo, conteo por ROI, ocupación y alertas."""
        e = self.escala
        px = lambda v: int(round(v * e))
        x0, y0, ancho, alto = px(6), px(6), px(292), px(20 + 17 * (len(monitores) + 1))
        capa = img.copy()
        cv2.rectangle(capa, (x0, y0), (x0 + ancho, y0 + alto), (20, 20, 20), -1)
        img = cv2.addWeighted(capa, 0.65, img, 0.35, 0)
        f, esc, gr = cv2.FONT_HERSHEY_SIMPLEX, 0.42 * e, max(1, round(e))
        cv2.putText(img, f"MONITOR VIAL   t = {t:5.1f} s", (x0 + px(6), y0 + px(15)), f, esc, (255, 255, 255), gr, cv2.LINE_AA)
        y = y0 + px(32)
        for m in monitores:
            cv2.rectangle(img, (x0 + px(6), y - px(9)), (x0 + px(16), y + px(1)), m.color, -1)
            cv2.putText(img, f"{m.nombre:<17} total:{len(m.contados):>3}  ahora:{m.ocupacion}", (x0 + px(22), y),
                        f, esc, (255, 255, 255), gr, cv2.LINE_AA)
            y += px(17)
        n_contra = sum(len(m.en_contravia) for m in monitores)
        cv2.putText(img, f"Contravias: {n_contra}   Alertas: {len(alertas)}", (x0 + px(6), y), f, esc,
                    (255, 255, 255) if n_contra == 0 else COLOR_ALERTA, gr, cv2.LINE_AA)
        return img

    def _banner(self, img, alertas, t):
        """Banner rojo inferior con la última alerta, visible durante 3 s tras producirse."""
        recientes = [a for a in alertas if 0 <= t - a["t_s"] <= 3.0]
        if not recientes:
            return img
        a, e = recientes[-1], self.escala
        texto = f"ALERTA: {a['detalle']} (#{a['id']})"
        esc, gr, m = 0.55 * e, max(2, round(2 * e)), int(10 * e)
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, esc, gr)
        x, y = (self.W - tw) // 2, self.H - int(40 * e)
        cv2.rectangle(img, (x - m, y - th - m), (x + tw + m, y + m), COLOR_ALERTA, -1)
        cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, esc, (255, 255, 255), gr, cv2.LINE_AA)
        return img

    def _anotar(self, frame, det, idx, monitores, alertas, clase_de):
        """Compone el frame de salida: ROI + trazas + cajas + etiquetas + marcos de alerta + panel + banner."""
        img = self.dibujar_rois(frame, alpha=0.22)
        if len(det):
            # Etiqueta de cada objeto: "#ID clase [permanencia s] [velocidad km/h]"
            etiquetas = []
            for tid in det.tracker_id:
                tid = int(tid)
                texto = f"#{tid} {clase_de(tid)}"
                perms = [p for p in (m.permanencia_actual(tid, idx) for m in monitores) if p is not None]
                if perms:
                    texto += f" {max(perms):.0f}s"
                vels = [m.velocidad_actual[tid] for m in monitores if tid in m.velocidad_actual and tid in m.visitas_activas]
                if vels:
                    texto += f" {vels[0]:.0f}km/h"
                etiquetas.append(texto)
            img = self.anot_trazas.annotate(img, det)
            img = self.anot_cajas.annotate(img, det)
            img = self.anot_etiquetas.annotate(img, det, labels=etiquetas)
            # Marco rojo extra alrededor de los objetos que dispararon alguna alerta.
            con_alerta = set().union(*[m.en_contravia | m.alerta_permanencia | m.alerta_velocidad for m in monitores])
            d = int(3 * self.escala)
            for (x1, y1, x2, y2), tid in zip(det.xyxy.astype(int), det.tracker_id):
                if int(tid) in con_alerta:
                    cv2.rectangle(img, (x1 - d, y1 - d), (x2 + d, y2 + d), COLOR_ALERTA, max(3, round(3 * self.escala)))
        img = self._panel(img, monitores, alertas, idx / self.fps)
        return self._banner(img, alertas, idx / self.fps)

    # ---------- pipeline ----------
    def procesar(self, ruta_salida, max_segundos=None, progreso=None):
        """Procesa el video. `progreso(fraccion, texto)` se llama periódicamente (para la barra de Streamlit)."""
        p = self.p
        fin = self.total_frames if not max_segundos else min(self.total_frames, int(max_segundos * self.fps))
        # ByteTrack asigna un ID persistente a cada objeto entre frames; un track perdido se
        # conserva ~2 s por si el objeto reaparece (oclusiones detrás de otros vehículos).
        tracker = sv.ByteTrack(frame_rate=self.fps_efectivo, lost_track_buffer=int(2 * self.fps_efectivo),
                               track_activation_threshold=0.25)
        monitores = [MonitorROI(k, cfg, self) for k, cfg in self.rois.items()]
        historial = defaultdict(lambda: deque(maxlen=p.ventana))   # Últimas posiciones (frame, x, y) por ID
        votos = defaultdict(Counter)                               # Votos de clase por ID
        alertas, serie = [], []
        # La clase de un ID es la más votada a lo largo de su vida: evita que un auto "parpadee" a camión.
        clase_de = lambda tid: CLASES[votos[tid].most_common(1)[0][0]] if votos[tid] else "?"

        cap = cv2.VideoCapture(self.ruta)
        # El video de salida solo contiene los frames procesados, por eso se escribe a fps_efectivo.
        info_salida = sv.VideoInfo(width=self.W, height=self.H, fps=self.fps_efectivo)
        idx = n = 0                                                # idx: frame leído · n: frames procesados
        with sv.VideoSink(target_path=ruta_salida, video_info=info_salida) as sink:
            while idx < fin:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % p.salto_frames == 0:
                    # 1) Detección YOLO (solo clases de interés) + filtro de conductores
                    res = self.modelo(frame, classes=list(CLASES), conf=p.conf, imgsz=p.imgsz, verbose=False)[0]
                    det = self.filtrar_conductores(sv.Detections.from_ultralytics(res))
                    # 2) Tracking: asocia las detecciones con IDs persistentes
                    det = tracker.update_with_detections(det)
                    # 3) Votación de clase e historial de posiciones (punto de apoyo en el suelo)
                    anclas = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(det) else []
                    for tid, cid, (x, y) in zip(det.tracker_id, det.class_id, anclas):
                        votos[int(tid)][int(cid)] += 1
                        historial[int(tid)].append((idx, float(x), float(y)))
                    if len(det):
                        # Sustituye la clase del frame por la clase estable (votada) de cada ID
                        det.class_id = np.array([votos[int(t)].most_common(1)[0][0] for t in det.tracker_id])
                    # 4) Lógica de cada ROI: conteo, permanencia, velocidad, contravía y alertas
                    for m in monitores:
                        m.actualizar(det, idx, clase_de, historial, alertas)
                    # 5) Serie temporal (acumulado y ocupación por ROI en cada frame procesado)
                    fila = {"frame": idx, "t_s": round(idx / self.fps, 3)}
                    for m in monitores:
                        fila[f"total_{m.clave}"], fila[f"ocupacion_{m.clave}"] = len(m.contados), m.ocupacion
                    serie.append(fila)
                    # 6) Frame anotado al video de salida
                    sink.write_frame(self._anotar(frame, det, idx, monitores, alertas, clase_de))
                    n += 1
                    if progreso and n % 10 == 0:
                        progreso(min(idx / max(fin, 1), 1.0),
                                 f"Frame {idx}/{fin} · contados " + " | ".join(f"{m.clave}:{len(m.contados)}" for m in monitores))
                idx += 1
        cap.release()
        for m in monitores:
            m.cerrar_todo(clase_de)
        if progreso:
            progreso(1.0, "Procesamiento terminado")
        return Resultados(self, monitores, alertas, pd.DataFrame(serie), votos, clase_de, n)


class Resultados:
    """Tablas finales del procesamiento (todas en pandas, listas para mostrar o exportar)."""

    def __init__(self, sistema, monitores, alertas, serie, votos, clase_de, frames_procesados):
        self.sistema, self.monitores, self.serie, self.frames_procesados = sistema, monitores, serie, frames_procesados
        self.visitas = (pd.DataFrame([v for m in monitores for v in m.visitas]).reindex(columns=COLUMNAS_VISITAS)
                        .sort_values(["t_entrada_s", "roi"]).reset_index(drop=True))
        self.alertas = pd.DataFrame(alertas, columns=["t_s", "frame", "tipo", "roi", "id", "clase", "detalle"])
        # Total de objetos: solo IDs vistos en ≥ k_persistencia frames (descarta tracks espurios de 1-2 frames).
        ids_validos = [t for t, v in votos.items() if sum(v.values()) >= sistema.p.k_persistencia]
        self.total_por_clase = pd.Series(Counter(clase_de(t) for t in ids_validos), dtype="int64").sort_values(ascending=False)
        self.total_objetos = len(ids_validos)

        # Tabla resumen: una fila por ROI activa.
        nombres = {m.clave: m.nombre for m in monitores}
        v = self.visitas
        self.resumen = pd.DataFrame({
            "ROI": [nombres[k] for k in nombres],
            "Objetos únicos": [v.loc[v.roi == k, "id"].nunique() for k in nombres],
            "Visitas": [int((v.roi == k).sum()) for k in nombres],
            "Permanencia media (s)": [round(v.loc[v.roi == k, "permanencia_s"].mean(), 2) if (v.roi == k).any() else np.nan for k in nombres],
            "Permanencia máx. (s)": [v.loc[v.roi == k, "permanencia_s"].max() if (v.roi == k).any() else np.nan for k in nombres],
            "Velocidad media (km/h)": [round(v.loc[v.roi == k, "velocidad_kmh"].mean(), 1) if v.loc[v.roi == k, "velocidad_kmh"].notna().any() else np.nan for k in nombres],
            "Contravías": [int(v.loc[v.roi == k, "contravia"].fillna(False).astype(bool).sum()) for k in nombres],
            "Alertas": [int((self.alertas.roi == k).sum()) for k in nombres],
        })
        # Conteo por clase × ROI con objetos únicos (un mismo ID que reingresa cuenta una sola vez).
        unicos = v.drop_duplicates(["roi", "id"])
        self.conteo_clase = (pd.crosstab(unicos["clase"], unicos["roi"]).rename(columns=nombres)
                             if len(unicos) else pd.DataFrame())
        if len(self.conteo_clase):
            self.conteo_clase["TOTAL"] = self.conteo_clase.sum(axis=1)
            self.conteo_clase = self.conteo_clase.sort_values("TOTAL", ascending=False)
        # Velocidades válidas: solo carril A (zona calibrada) y solo vehículos.
        self.velocidades = v[(v.roi == "A") & v.velocidad_kmh.notna() & (v.clase != "persona")]

    def estadisticas(self):
        """KPIs globales que la interfaz muestra en la fila de métricas."""
        v = self.visitas
        return {
            "total_objetos": self.total_objetos,
            "permanencia_media_s": round(float(v.permanencia_s.mean()), 2) if len(v) else float("nan"),
            "visitas_totales": int(len(v)),
            "velocidad_media_kmh": round(float(self.velocidades.velocidad_kmh.mean()), 1) if len(self.velocidades) else float("nan"),
            "contravias": int(sum(len(m.en_contravia) for m in self.monitores)),
            "alertas": int(len(self.alertas)),
        }

    def a_excel(self):
        """Devuelve un Excel en memoria con una hoja por tabla."""
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xls:
            self.resumen.to_excel(xls, sheet_name="Resumen por ROI", index=False)
            if len(self.conteo_clase):
                self.conteo_clase.to_excel(xls, sheet_name="Conteo por clase")
            self.visitas.to_excel(xls, sheet_name="Visitas", index=False)
            self.alertas.to_excel(xls, sheet_name="Alertas", index=False)
            self.velocidades.to_excel(xls, sheet_name="Velocidades carril A", index=False)
        return buf.getvalue()


def convertir_h264(origen, destino):
    """Convierte el video (mp4v) a H.264 para que el navegador lo reproduzca. Devuelve la ruta usable."""
    # Usa el ffmpeg del sistema; si no hay, el binario que trae imageio-ffmpeg; si tampoco, devuelve el original.
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return origen
    r = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", origen, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", destino])
    return destino if r.returncode == 0 and os.path.exists(destino) else origen
