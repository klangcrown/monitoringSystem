"""
Система мониторинга дисциплины на занятиях — Streamlit интерфейс
Часть 2: Графический интерфейс
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Отрисовка текста с поддержкой Unicode (кириллица) через PIL
# ──────────────────────────────────────────────────────────────────────────────
def _cv2_put_text_unicode(frame, text, pos, font_size=18, color=(255,255,255), bg_color=None):
    """Рисует текст с кириллицей на кадре через PIL."""
    try:
        from PIL import ImageFont, ImageDraw, Image
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # Пробуем загрузить шрифт с поддержкой кириллицы
        font = None
        for font_path in [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/times.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()

        x, y = pos
        if bg_color is not None:
            # Фон под текстом
            bbox = draw.textbbox((x, y), text, font=font)
            draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=bg_color)
        pil_color = (color[2], color[1], color[0])  # BGR -> RGB
        draw.text((x, y), text, font=font, fill=pil_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback: cv2 (без кириллицы)
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame


st.set_page_config(
    page_title="Мониторинг дисциплины",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main, .stApp { background-color: #0f1117; }
    h1, h2, h3 { color: #e0e0e0 !important; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3e);
        border: 1px solid #3a4060; border-radius: 12px;
        padding: 16px 20px; margin: 6px 0; text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #7eb8f7; line-height: 1; }
    .metric-label { font-size: 0.8rem; color: #8a9bc0; margin-top: 4px;
                    text-transform: uppercase; letter-spacing: 0.08em; }
    .violation-row {
        background: #1a1f2e; border-left: 3px solid #e05c5c;
        border-radius: 6px; padding: 10px 14px; margin: 5px 0;
        font-size: 0.85rem; color: #c8d0e0;
    }
    .violation-row.phone  { border-left-color: #f0a500; }
    .violation-row.sleep  { border-left-color: #5c85e0; }
    .violation-row.bottle { border-left-color: #50c878; }
    .violation-row.food   { border-left-color: #d45ce0; }
    section[data-testid="stSidebar"] { background: #12151f; border-right: 1px solid #2a2f45; }
    .stButton > button {
        background: linear-gradient(135deg, #2c6fad, #1a4a7a);
        color: white; border: none; border-radius: 8px; font-weight: 600;
    }
    .video-info { background: #1a1f2e; border-radius: 8px; padding: 10px 14px;
                  color: #8a9bc0; font-size: 0.82rem; margin: 6px 0; }
    .log-container { max-height: 420px; overflow-y: auto; padding-right: 4px; }
    .log-container::-webkit-scrollbar { width: 4px; }
    .log-container::-webkit-scrollbar-track { background: #1a1f2e; }
    .log-container::-webkit-scrollbar-thumb { background: #3a4060; border-radius: 2px; }
    hr { border-color: #2a2f45; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

VIOLATION_COLORS = {
    "phone_usage": (0, 165, 255),
    "sleeping":    (220, 80,  80),
    "bottle":      (80,  200, 120),
    "food":        (200, 80,  220),
}
VIOLATION_LABELS = {
    "phone_usage": "Телефон",
    "sleeping":    "Сон",
    "bottle":      "Бутылка",
    "food":        "Еда",
}
CV2_LABELS = {
    "phone_usage": "[!] Phone",
    "sleeping":    "[!] Sleep",
    "bottle":      "[!] Bottle",
    "food":        "[!] Food",
}
VIOLATION_CSS = {
    "phone_usage": "phone",
    "sleeping":    "sleep",
    "bottle":      "bottle",
    "food":        "food",
}
VIOLATION_EMOJI = {
    "phone_usage": "📱",
    "sleeping":    "😴",
    "bottle":      "🥤",
    "food":        "🍔",
}
PHONE_CLASSES  = [67]
BOTTLE_CLASSES = [39]
FOOD_CLASSES   = list(range(46, 56))

_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]
EAR_THRESHOLD   = 0.22
SLEEP_TRIGGER_S = 3.0


# ──────────────────────────────────────────────────────────────────────────────
# Запись видеосегментов нарушений
# ──────────────────────────────────────────────────────────────────────────────
class SegmentRecorder:
    """Пишет видео в кольцевой буфер и сохраняет сегмент при нарушении."""
    def __init__(self, fps=15, buffer_sec=5, out_dir="outputs/segments"):
        self.fps        = fps
        self.buf_frames = int(fps * buffer_sec)
        self.out_dir    = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._buf   = []          # кольцевой буфер сырых кадров
        self._writers: dict = {}  # vtype -> (VideoWriter, path, start_time)
        self._lock  = threading.Lock()

    def push(self, frame):
        """Добавляет кадр в кольцевой буфер."""
        with self._lock:
            self._buf.append(frame.copy())
            if len(self._buf) > self.buf_frames:
                self._buf.pop(0)

    def start_violation(self, vtype: str):
        """Открывает VideoWriter для данного типа нарушения (если ещё не открыт)."""
        with self._lock:
            if vtype in self._writers:
                return
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.out_dir, f"{vtype}_{ts}.mp4")
            h, w = self._buf[0].shape[:2] if self._buf else (480, 640)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
            # Дозаписываем буфер (предысторию)
            for f in self._buf:
                writer.write(f)
            self._writers[vtype] = (writer, path, datetime.now())

    def write_violation(self, vtype: str, frame):
        """Дописывает кадр в активный сегмент."""
        with self._lock:
            if vtype in self._writers:
                self._writers[vtype][0].write(frame)

    def stop_violation(self, vtype: str) -> str | None:
        """Закрывает VideoWriter, возвращает путь к файлу."""
        with self._lock:
            if vtype not in self._writers:
                return None
            writer, path, _ = self._writers.pop(vtype)
            writer.release()
            return path

    def stop_all(self) -> list:
        """Закрывает все активные сегменты."""
        with self._lock:
            paths = []
            for vtype, (writer, path, _) in list(self._writers.items()):
                writer.release()
                paths.append(path)
            self._writers.clear()
            return paths


def _save_face_photo(frame, box, name: str, out_dir: str = "outputs/faces") -> str:
    """Вырезает и сохраняет фото лица нарушителя (включая Unknown)."""
    os.makedirs(out_dir, exist_ok=True)
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    pad = 20
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        face_img = frame
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Для Unknown добавляем микросекунды чтобы не было конфликтов имён
    safe = name.replace(" ", "_").replace("/", "_") if name != "Unknown" else f"Unknown_{ts[-6:]}"
    path = os.path.join(out_dir, f"{safe}_{ts[:15]}.jpg")
    cv2.imwrite(path, face_img)
    return path


def _save_report(violations: list, out_dir: str = "outputs/reports"):
    """Сохраняет JSON + TXT отчёт."""
    if not violations:
        return
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    counts = defaultdict(int)
    for v in violations:
        counts[v["type"]] += 1

    report = {
        "date":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total":      len(violations),
        "counts":     dict(counts),
        "violations": violations,
    }
    json_path = os.path.join(out_dir, f"report_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(out_dir, f"report_{ts}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"ОТЧЁТ О НАРУШЕНИЯХ\n{'='*40}\n")
        f.write(f"Дата: {report['date']}\n")
        f.write(f"Всего нарушений: {report['total']}\n\n")
        for vtype, cnt in counts.items():
            f.write(f"  {VIOLATION_LABELS.get(vtype, vtype)}: {cnt}\n")
        f.write("\nДЕТАЛИ:\n")
        for v in violations:
            lbl  = VIOLATION_LABELS.get(v["type"], v["type"])
            face = v.get("person", "—")
            photo = v.get("face_photo", "")
            seg   = v.get("segment", "")
            f.write(f"  [{v['start']} – {v['end']}] {lbl} | {v['duration']} сек"
                    f" | {v['conf']:.0%} | {face}")
            if photo: f.write(f" | фото: {photo}")
            if seg:   f.write(f" | видео: {seg}")
            f.write("\n")


@st.cache_resource(show_spinner=False)
def _load_faces_db(db_path: str = "database/faces_db.pkl"):
    """Загружает базу лиц. Возвращает {name: encoding} или {}."""
    try:
        import pickle, os
        if not os.path.exists(db_path):
            return {}
        with open(db_path, "rb") as f:
            db = pickle.load(f)
        # Формат: {name: {"encoding": np.array, ...}} или {name: np.array}
        result = {}
        for name, val in db.items():
            if isinstance(val, dict) and "encoding" in val:
                result[name] = val["encoding"]
            elif hasattr(val, "__len__") and len(val) == 128:
                result[name] = val
        return result
    except Exception:
        return {}


# Глобальный каскадный детектор (быстрый, синхронный — только для bbox)
_haar_cascade = None
def _get_haar():
    global _haar_cascade
    if _haar_cascade is None:
        for path in ["haarcascade_frontalface_default.xml",
                     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"]:
            if os.path.exists(path):
                _haar_cascade = cv2.CascadeClassifier(path)
                break
    return _haar_cascade


def _fast_face_boxes(frame):
    """Быстрое определение bbox лиц через Haar (без распознавания)."""
    cascade = _get_haar()
    if cascade is None:
        return []
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    result = []
    for (x, y, w, h) in faces:
        result.append([x*2, y*2, (x+w)*2, (y+h)*2])  # масштабируем обратно
    return result


def _recognize_faces(frame, faces_db):
    """Распознавание лиц — запускается в отдельном потоке."""
    if not faces_db:
        return []
    try:
        import face_recognition as fr
        # Уменьшаем кадр в 2 раза для ускорения
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(rgb, model="hog")
        if not locations:
            return []
        encodings = fr.face_encodings(rgb, locations)
        results = []
        known_names = list(faces_db.keys())
        known_encs  = list(faces_db.values())
        for (top, right, bottom, left), enc in zip(locations, encodings):
            name = "Unknown"
            known = False
            if known_encs:
                distances = fr.face_distance(known_encs, enc)
                best_idx  = int(np.argmin(distances))
                if distances[best_idx] < 0.55:
                    name  = known_names[best_idx]
                    known = True
            # Масштабируем координаты обратно
            results.append({"name": name,
                             "box": [left*2, top*2, right*2, bottom*2],
                             "known": known})
        return results
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def _load_face_mesh():
    """
    Загружает MediaPipe FaceMesh.
    Обходит проблему кириллицы в пути — патчим os.path.exists и os.path.isfile
    чтобы MediaPipe мог найти свои внутренние файлы моделей.
    """
    try:
        import mediapipe as mp
        import os
        import builtins

        # Патч: подменяем open для чтения бинарных файлов моделей mediapipe
        _orig_open = builtins.open
        _orig_exists = os.path.exists
        _orig_isfile = os.path.isfile

        def _patched_open(file, mode='r', *args, **kwargs):
            if isinstance(file, str) and 'mediapipe' in file.lower() and 'b' in str(mode):
                try:
                    return _orig_open(file, mode, *args, **kwargs)
                except (OSError, FileNotFoundError):
                    # Пробуем через короткие пути (8.3 формат Windows)
                    import ctypes
                    buf = ctypes.create_unicode_buffer(512)
                    ctypes.windll.kernel32.GetShortPathNameW(file, buf, 512)
                    short = buf.value
                    if short:
                        return _orig_open(short, mode, *args, **kwargs)
            return _orig_open(file, mode, *args, **kwargs)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        test = np.zeros((64, 64, 3), dtype=np.uint8)
        face_mesh.process(test)
        return face_mesh, None
    except Exception as e:
        # Второй способ: через mediapipe Tasks API с байтами модели
        try:
            import mediapipe as mp
            import importlib.resources as pkg_resources

            # Ищем face_landmarker.task — пробуем несколько мест
            import ctypes, sys
            _app_dir = Path(sys.argv[0]).parent.resolve() if sys.argv else Path.cwd()
            _candidates = [
                _app_dir / "face_landmarker.task",
                Path.cwd() / "face_landmarker.task",
                Path(__file__).parent / "face_landmarker.task",
            ]
            task_path = None
            for _p in _candidates:
                if _p.exists():
                    task_path = _p
                    break

            if task_path is None:
                return None, (f"face_landmarker.task не найден в {_app_dir}. "
                              f"Ошибка FaceMesh: {e}")

            # Читаем байты через короткий путь Windows (8.3) — обход кириллицы
            buf = ctypes.create_unicode_buffer(512)
            ctypes.windll.kernel32.GetShortPathNameW(str(task_path), buf, 512)
            short_path = buf.value or str(task_path)
            with open(short_path, "rb") as f:
                model_data = f.read()

            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_opts = mp_python.BaseOptions(model_asset_buffer=model_data)
            opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                num_faces=4,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
            landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
            return ("tasks", landmarker), None
        except Exception as e2:
            return None, f"FaceMesh: {e} | Tasks: {e2}" 


@st.cache_resource(show_spinner=False)
def load_yolo(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path), None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner=False)
def load_sleep_model(model_path: str):
    """Загружает модель для детекции сна. Возвращает (model, sleep_class_id, err)."""
    if not model_path.strip():
        return None, None, "no path"
    try:
        from ultralytics import YOLO
        m = YOLO(model_path)
        # Ищем класс сна по имени
        sleep_id = None
        for cls_id, cls_name in m.names.items():
            if cls_name.lower() in ("sleep", "sleeping", "sleepy", "сон"):
                sleep_id = cls_id
                break
        # Если явного класса нет — берём первый класс (у кастомных моделей часто 1 класс)
        if sleep_id is None and len(m.names) <= 5:
            sleep_id = list(m.names.keys())[0]
        return m, sleep_id, None
    except Exception as e:
        return None, None, str(e)


def _ear(landmarks, indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0.3


class ViolationTracker:
    def __init__(self, buffer_sec=5.0, min_sec=2.0):
        self.buffer_sec = buffer_sec
        self.min_sec    = min_sec
        self._active: dict = {}
        self._done:   list = []
        self._lock = threading.Lock()

    def update(self, detected: list, now: float):
        with self._lock:
            seen = set()
            for d in detected:
                key = d["type"]
                seen.add(key)
                if key not in self._active:
                    self._active[key] = {"type": d["type"], "conf": d["conf"],
                                          "start": now, "last_seen": now}
                else:
                    self._active[key]["last_seen"] = now
                    if d["conf"] > self._active[key]["conf"]:
                        self._active[key]["conf"] = d["conf"]
            for key in list(self._active):
                if key not in seen and now - self._active[key]["last_seen"] >= self.buffer_sec:
                    info = self._active.pop(key)
                    dur = info["last_seen"] - info["start"]
                    if dur >= self.min_sec:
                        self._done.append({
                            "type":     info["type"],
                            "conf":     info["conf"],
                            "start":    datetime.fromtimestamp(info["start"]).strftime("%H:%M:%S"),
                            "end":      datetime.fromtimestamp(info["last_seen"]).strftime("%H:%M:%S"),
                            "duration": int(dur),
                            "person":   "—",
                        })

    def flush(self, now: float):
        with self._lock:
            for key, info in list(self._active.items()):
                dur = now - info["start"]
                if dur >= self.min_sec:
                    self._done.append({
                        "type":     info["type"],
                        "conf":     info["conf"],
                        "start":    datetime.fromtimestamp(info["start"]).strftime("%H:%M:%S"),
                        "end":      datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                        "duration": int(dur),
                        "person":   "—",
                    })
            self._active.clear()

    def pop_done(self):
        with self._lock:
            done = list(self._done)
            self._done.clear()
            return done

    def active_types(self):
        with self._lock:
            return list(self._active.keys())


class VideoStream:
    def __init__(self, source, model, face_mesh, conf, buffer_sec, min_sec,
                 frame_skip, enabled_types, sleep_model=None, sleep_class_id=None,
                 ear_threshold=0.22, sleep_trigger=3.0):
        self.source         = source
        self.model          = model
        self.face_mesh      = face_mesh
        self.sleep_model    = sleep_model
        self.sleep_class_id = sleep_class_id
        self.ear_threshold  = ear_threshold
        self.sleep_trigger  = sleep_trigger
        self.conf          = conf
        self.frame_skip    = frame_skip
        self.enabled_types = enabled_types
        self.tracker       = ViolationTracker(buffer_sec, min_sec)
        self._stop         = threading.Event()
        self._frame_lock   = threading.Lock()
        self._last_frame   = None
        self._new_violations: list = []
        self._viol_lock    = threading.Lock()
        self._eye_closed_since: dict = {}
        self._tmp_path        = None
        self._faces_db        = {}   # заполняется через set_faces_db()
        self._face_cache      = []   # кэш: [{"name","box","known"}]
        self._face_cache_idx  = 0    # счётчик кадров для обновления кэша
        self._face_recog_every = 10  # обновлять распознавание каждые N кадров
        self._face_thread     = None # поток распознавания лиц
        self._face_thread_lock = threading.Lock()
        self._recorder        = SegmentRecorder(fps=15, buffer_sec=buffer_sec)
        self._active_vtypes   = set()   # типы нарушений активные прямо сейчас
        self._thread       = threading.Thread(target=self._run, daemon=True)

    def set_faces_db(self, db: dict):
        self._faces_db = db

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_alive(self):
        return self._thread.is_alive()

    def get_frame(self):
        with self._frame_lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def pop_violations(self):
        with self._viol_lock:
            v = list(self._new_violations)
            self._new_violations.clear()
            return v

    def _detect_yolo(self, frame):
        detected = []
        if self.model is None:
            return frame, detected
        try:
            results = self.model(frame, conf=self.conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    vtype = None
                    if "phone_usage" in self.enabled_types and cls_id in PHONE_CLASSES:
                        vtype = "phone_usage"
                    elif "bottle" in self.enabled_types and cls_id in BOTTLE_CLASSES:
                        vtype = "bottle"
                    elif "food" in self.enabled_types and cls_id in FOOD_CLASSES:
                        vtype = "food"
                    if vtype is None:
                        continue
                    color = VIOLATION_COLORS[vtype]
                    label = f"{CV2_LABELS[vtype]} {conf:.0%}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), color, -1)
                    cv2.putText(frame, label, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    detected.append({"type": vtype, "conf": conf, "box": [x1, y1, x2, y2]})
        except Exception:
            pass
        return frame, detected

    def _get_landmarks(self, frame):
        """Возвращает список landmarks для каждого лица. Поддерживает FaceMesh и Tasks API."""
        if self.face_mesh is None:
            return []
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tasks API режим
        if isinstance(self.face_mesh, tuple) and self.face_mesh[0] == "tasks":
            import mediapipe as mp
            landmarker = self.face_mesh[1]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            # Конвертируем Tasks landmarks в формат совместимый с FaceMesh
            all_lm = []
            for face_lm in (result.face_landmarks or []):
                all_lm.append(face_lm)
            return all_lm

        # Legacy FaceMesh режим
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return []
        return [fl.landmark for fl in results.multi_face_landmarks]

    def _detect_sleep(self, frame, now):
        """Детекция сна через MediaPipe EAR (Eye Aspect Ratio)."""
        detected = []
        if "sleeping" not in self.enabled_types or self.face_mesh is None:
            return frame, detected
        try:
            h, w = frame.shape[:2]
            all_landmarks = self._get_landmarks(frame)
            if not all_landmarks:
                return frame, detected
            for face_idx, lm in enumerate(all_landmarks):
                ear_l = _ear(lm, _LEFT_EYE,  w, h)
                ear_r = _ear(lm, _RIGHT_EYE, w, h)
                ear   = (ear_l + ear_r) / 2.0
                eyes_closed = ear < self.ear_threshold
                if eyes_closed:
                    if face_idx not in self._eye_closed_since:
                        self._eye_closed_since[face_idx] = now
                    elapsed = now - self._eye_closed_since[face_idx]
                else:
                    self._eye_closed_since.pop(face_idx, None)
                    elapsed = 0.0
                # Bbox только по точкам глаз (не всего лица)
                eye_indices = _LEFT_EYE + _RIGHT_EYE
                xs = [int(lm[i].x * w) for i in eye_indices]
                ys = [int(lm[i].y * h) for i in eye_indices]
                pad = 8
                x1 = max(0, min(xs)-pad); y1 = max(0, min(ys)-pad)
                x2 = min(w, max(xs)+pad); y2 = min(h, max(ys)+pad)
                # Для detected box — всё лицо (для трекера нарушений)
                face_xs = [int(p.x * w) for p in lm]
                face_ys = [int(p.y * h) for p in lm]
                fx1 = max(0, min(face_xs)-10); fy1 = max(0, min(face_ys)-10)
                fx2 = min(w, max(face_xs)+10); fy2 = min(h, max(face_ys)+10)
                if elapsed >= self.sleep_trigger:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 80, 80), 2)
                    lbl_s = f"[!] Sleep {elapsed:.0f}s"
                    (tw, th), _ = cv2.getTextSize(lbl_s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+6, y1), (220, 80, 80), -1)
                    cv2.putText(frame, lbl_s, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    detected.append({"type": "sleeping", "conf": 0.9,
                                     "box": [fx1, fy1, fx2, fy2]})
                else:
                    # Глаза открыты — показываем EAR для отладки
                    cv2.putText(frame, f"EAR:{ear:.2f}", (x1+3, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 180, 100), 1)
                if eyes_closed and elapsed < self.sleep_trigger:
                    # Глаза закрыты но триггер ещё не сработал — жёлтая рамка
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 1)
                    lbl_w = f"Eyes closed {elapsed:.1f}s/{self.sleep_trigger:.0f}s"
                    cv2.putText(frame, lbl_w, (x1+3, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)
        except Exception:
            pass
        return frame, detected

    def _detect_faces(self, frame):
        """Обновляет кэш лиц в фоновом потоке, рисует рамки без блокировки."""
        self._face_cache_idx += 1
        if self._face_cache_idx % self._face_recog_every == 0:
            # Запускаем распознавание в отдельном потоке — не блокируем видеопоток
            with self._face_thread_lock:
                if self._face_thread is None or not self._face_thread.is_alive():
                    frame_copy = frame.copy()
                    def _run_recog(f, db):
                        result = _recognize_faces(f, db)
                        self._face_cache = result
                    self._face_thread = threading.Thread(
                        target=_run_recog, args=(frame_copy, self._faces_db), daemon=True)
                    self._face_thread.start()

        for face in self._face_cache:
            x1, y1, x2, y2 = face["box"]
            if face["known"]:
                color = (180, 0, 180)   # фиолетовый BGR — известный
                lbl   = face["name"]
                text_color = (255, 255, 255)
            else:
                color = (40, 40, 40)    # тёмный — неизвестный
                lbl   = "Unknown"
                text_color = (200, 200, 200)
            # Рамка вокруг лица
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Фон под подписью
            cv2.rectangle(frame, (x1, y1-26), (x1+len(lbl)*10+6, y1), color, -1)
            # Текст через PIL (поддержка кириллицы)
            frame = _cv2_put_text_unicode(frame, lbl, (x1+3, y1-22),
                                          font_size=16, color=text_color)
        return frame

    def _run(self):
        if isinstance(self.source, int):
            cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            return

        frame_idx = 0
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Кольцевой буфер — толкаем каждый кадр (до детекции)
            self._recorder.push(frame)

            if frame_idx % self.frame_skip != 0:
                continue

            now = time.time()
            annotated = frame.copy()

            annotated, detected = self._detect_yolo(annotated)
            annotated, sleep_det = self._detect_sleep(annotated, now)
            detected.extend(sleep_det)
            annotated = self._detect_faces(annotated)

            # Определяем активные типы нарушений прямо сейчас
            current_vtypes = {d["type"] for d in detected}

            # Начинаем запись новых нарушений
            for vtype in current_vtypes - self._active_vtypes:
                self._recorder.start_violation(vtype)

            # Пишем кадр во все активные сегменты
            for vtype in current_vtypes:
                self._recorder.write_violation(vtype, frame)

            # Завершаем запись нарушений которые исчезли
            finished_paths = {}
            for vtype in self._active_vtypes - current_vtypes:
                path = self._recorder.stop_violation(vtype)
                if path:
                    finished_paths[vtype] = path

            self._active_vtypes = current_vtypes

            # Обновляем трекер
            self.tracker.update(detected, now)
            new_v = self.tracker.pop_done()

            # Обогащаем нарушения: имя нарушителя, фото лица, путь к видео
            if new_v:
                for v in new_v:
                    # Имя из кэша лиц — берём любое лицо (известное или нет)
                    if self._face_cache:
                        known_faces = [f for f in self._face_cache if f["known"]]
                        best_face   = known_faces[0] if known_faces else self._face_cache[0]
                        v["person"]   = best_face["name"]
                        v["face_box"] = best_face["box"]
                    else:
                        v["person"] = "Unknown"
                        v["face_box"] = None
                    # Фото лица — если bbox из кэша нет, ищем через Haar (быстро)
                    try:
                        face_box = v.pop("face_box", None)
                        if face_box is None:
                            # Haar — быстрый синхронный fallback
                            boxes = _fast_face_boxes(frame)
                            if boxes:
                                face_box = boxes[0]
                        if face_box is None and detected:
                            d = next((d for d in detected if d["type"] == v["type"]), detected[0])
                            face_box = d.get("box")
                        if face_box:
                            photo_path = _save_face_photo(frame, face_box, v["person"])
                            v["face_photo"] = photo_path
                    except Exception:
                        pass
                    # Путь к видеосегменту
                    if v["type"] in finished_paths:
                        v["segment"] = finished_paths[v["type"]]

                with self._viol_lock:
                    self._new_violations.extend(new_v)

            ts = datetime.now().strftime("%H:%M:%S")
            h, w = annotated.shape[:2]
            cv2.putText(annotated, ts, (w-120, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 200, 255), 1)
            y_pos = 30
            for vtype in self.tracker.active_types():
                lbl   = CV2_LABELS.get(vtype, vtype)
                color = VIOLATION_COLORS.get(vtype, (200, 200, 200))
                cv2.putText(annotated, lbl, (8, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 26

            with self._frame_lock:
                self._last_frame = annotated

        # Завершение — закрываем все открытые сегменты
        self._recorder.stop_all()
        self.tracker.flush(time.time())
        new_v = self.tracker.pop_done()
        if new_v:
            for v in new_v:
                if self._face_cache:
                    known_faces = [f for f in self._face_cache if f["known"]]
                    best_face   = known_faces[0] if known_faces else self._face_cache[0]
                    v["person"] = best_face["name"]
                else:
                    v["person"] = "—"
            with self._viol_lock:
                self._new_violations.extend(new_v)
        cap.release()


def init_state():
    for k, v in [("violations", []), ("video_stream", None), ("model_path", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def journal_html(violations):
    counts = defaultdict(int)
    for v in violations:
        counts[v["type"]] += 1
    total = len(violations)

    cards = "<div style='display:flex;gap:8px;margin-bottom:10px;'>"
    for vtype in ["phone_usage", "sleeping", "bottle", "food"]:
        emoji = VIOLATION_EMOJI[vtype]
        cards += (f"<div class='metric-card' style='flex:1'>"
                  f"<div class='metric-value'>{counts[vtype]}</div>"
                  f"<div class='metric-label'>{emoji}</div></div>")
    cards += "</div>"
    cards += (f"<p style='color:#8a9bc0;font-size:.85rem;margin:4px 0 10px'>"
              f"Всего нарушений: <b style='color:#e0e0e0'>{total}</b></p>")

    if not violations:
        return cards + "<p style='color:#8a9bc0'>Нарушений пока не зафиксировано.</p>"

    rows = ""
    for v in reversed(violations[-50:]):
        css  = VIOLATION_CSS.get(v["type"], "")
        lbl  = VIOLATION_EMOJI.get(v["type"], "") + " " + VIOLATION_LABELS.get(v["type"], v["type"])
        extra = ""
        if v.get("face_photo"):
            extra += f" | 📸 <code>{os.path.basename(v['face_photo'])}</code>"
        if v.get("segment"):
            extra += f" | 🎬 <code>{os.path.basename(v['segment'])}</code>"
        rows += (f"<div class='violation-row {css}'>"
                 f"<b>{v['start']} – {v['end']}</b>"
                 f" | {lbl} | {v['duration']} сек"
                 f" | <i>{v['conf']:.0%}</i>"
                 f" | 👤 {v.get('person','—')}"
                 f"{extra}</div>")
    return cards + f"<div class='log-container'>{rows}</div>"


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Настройки")
        st.markdown("---")
        st.markdown("### 📡 Источник видео")
        source = st.radio("Источник", ["Веб-камера", "Видеофайл", "URL-поток"],
                          label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 🤖 Модель YOLO")
        model_path = st.text_input("Путь к модели (.pt)", value="yolo11n.pt",
                                    help="YOLO модель для детекции телефона, бутылки, еды (COCO). Сон определяется через MediaPipe.")
        sleep_model_path = ""  # не используется
        st.markdown("---")
        st.markdown("### 🔧 Параметры детекции")
        conf       = st.slider("Порог уверенности", 0.1, 1.0, 0.5, 0.05)
        buffer_sec = st.slider("Буфер записи (сек)", 1, 30, 5)
        min_sec    = st.slider("Мин. длительность (сек)", 1, 15, 2)
        frame_skip = st.slider("Пропуск кадров", 1, 8, 3)
        st.markdown("---")
        st.markdown("### 😴 Параметры детекции сна")
        ear_threshold = st.slider("Порог EAR (закрытые глаза)",
                                   min_value=0.10, max_value=0.35, value=0.22, step=0.01,
                                   help="Чем ниже — тем сильнее нужно закрыть глаза. "
                                        "При очках обычно 0.18–0.20")
        sleep_trigger = st.slider("Время до фиксации сна (сек)",
                                   min_value=1, max_value=10, value=3,
                                   help="Сколько секунд глаза должны быть закрыты")
        st.markdown("### 🚨 Типы нарушений")
        en_phone  = st.checkbox("📱 Телефон",  value=True)
        en_sleep  = st.checkbox("😴 Сон",      value=True)
        en_bottle = st.checkbox("🥤 Бутылка",  value=True)
        en_food   = st.checkbox("🍔 Еда",      value=True)
        enabled = set()
        if en_phone:  enabled.add("phone_usage")
        if en_sleep:  enabled.add("sleeping")
        if en_bottle: enabled.add("bottle")
        if en_food:   enabled.add("food")
        st.markdown("---")
        if st.button("🗑️ Очистить журнал", width='stretch'):
            st.session_state.violations = []
            st.rerun()
    return {"source": source, "model_path": model_path,
            "sleep_model_path": sleep_model_path,
            "conf": conf, "buffer_sec": buffer_sec, "min_sec": min_sec,
            "frame_skip": frame_skip, "enabled": enabled,
            "ear_threshold": ear_threshold, "sleep_trigger": sleep_trigger}


def main():
    st.markdown("# 📹 Система мониторинга дисциплины")
    st.markdown("Автоматическое обнаружение нарушений: сон, телефон, еда/напитки")
    st.markdown("---")

    cfg = render_sidebar()

    if st.session_state.model_path != cfg["model_path"]:
        st.session_state.model_path = cfg["model_path"]
        load_yolo.clear()

    model_status = st.empty()
    model, err = load_yolo(cfg["model_path"])
    if err:
        model_status.error(f"Ошибка основной модели: {err}")
    else:
        model_status.empty()

    # Sleep детектируется через MediaPipe EAR (без отдельной модели)
    sleep_model, sleep_class_id = None, None

    # База лиц
    faces_db = _load_faces_db()
    if faces_db:
        st.sidebar.success(f"👤 База лиц: {len(faces_db)} чел. ({', '.join(faces_db.keys())})")
    else:
        st.sidebar.info("👤 База лиц пуста — все лица будут 'Unknown'")

    # Очищаем кэш при первом запуске (чтобы подхватить новую версию функции)
    if "face_mesh_initialized" not in st.session_state:
        _load_face_mesh.clear()
        st.session_state["face_mesh_initialized"] = True

    face_mesh, face_mesh_err = _load_face_mesh()
    if face_mesh_err:
        st.warning(f"⚠️ MediaPipe FaceMesh не загружен: {face_mesh_err}. Детекция сна недоступна.")
    else:
        pass  # FaceMesh загружен успешно

    vs: VideoStream = st.session_state.video_stream
    is_running = vs is not None and vs.is_alive()

    col_video, col_journal = st.columns([3, 2], gap="large")

    with col_journal:
        st.markdown("### 📊 Журнал нарушений")
        journal_ph  = st.empty()
        download_ph = st.empty()

    with col_video:
        st.markdown("### 🎥 Видеопоток")
        source_name = cfg["source"]

        btn1, btn2 = st.columns(2)
        with btn1:
            start_label = {"Веб-камера": "▶️ Запустить камеру",
                           "Видеофайл":  "▶️ Запустить",
                           "URL-поток":  "▶️ Подключиться"}[source_name]
            start_btn = st.button(start_label, disabled=is_running,
                                  width='stretch')
        with btn2:
            stop_btn = st.button("⏹️ Остановить", disabled=not is_running,
                                 width='stretch')

        upload_file = None
        url_input   = ""
        cam_index   = 0
        if source_name == "Веб-камера":
            cam_index = st.number_input("Индекс камеры", 0, 5, 0, 1)
        elif source_name == "Видеофайл":
            upload_file = st.file_uploader(
                "Видеофайл", type=["mp4","avi","mov","mkv","webm"],
                label_visibility="collapsed")
        elif source_name == "URL-поток":
            url_input = st.text_input("URL потока",
                placeholder="rtsp://... или http://...")

        frame_ph = st.empty()

        if start_btn and not is_running:
            source_val = None
            tmp_path   = None
            if source_name == "Веб-камера":
                source_val = int(cam_index)
            elif source_name == "Видеофайл" and upload_file:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(upload_file.name).suffix)
                tmp.write(upload_file.read())
                tmp.close()
                tmp_path   = tmp.name
                source_val = tmp_path
            elif source_name == "URL-поток" and url_input.strip():
                source_val = url_input.strip()

            if source_val is not None:
                new_vs = VideoStream(
                    source=source_val, model=model, face_mesh=face_mesh,
                    conf=cfg["conf"], buffer_sec=cfg["buffer_sec"],
                    min_sec=cfg["min_sec"], frame_skip=cfg["frame_skip"],
                    enabled_types=cfg["enabled"],
                    sleep_model=sleep_model,
                    sleep_class_id=sleep_class_id,
                    ear_threshold=cfg.get("ear_threshold", 0.22),
                    sleep_trigger=float(cfg.get("sleep_trigger", 3.0)),
                )
                new_vs._tmp_path = tmp_path
                new_vs.set_faces_db(faces_db)
                new_vs.start()
                st.session_state.video_stream = new_vs
                st.rerun()

        if stop_btn and is_running:
            vs.stop()
            vs._thread.join(timeout=2.0)
            new_v = vs.pop_violations()
            if new_v:
                st.session_state.violations.extend(new_v)
            # Автосохранение отчёта
            _save_report(st.session_state.violations)
            tmp = getattr(vs, "_tmp_path", None)
            if tmp and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except Exception:
                    pass
            st.session_state.video_stream = None
            st.rerun()

        # Фрагмент обновляется каждые 200мс — без перерисовки всей страницы
        @st.fragment(run_every=0.2)
        def _live_view():
            _vs  = st.session_state.video_stream
            _run = _vs is not None and _vs.is_alive()

            if _run:
                # Забираем новые нарушения из потока
                new_v = _vs.pop_violations()
                if new_v:
                    st.session_state.violations.extend(new_v)

                frm = _vs.get_frame()
                if frm is not None:
                    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                    frame_ph.image(rgb, channels="RGB", width='stretch')
                else:
                    frame_ph.markdown(
                        '<div class="video-info">⏳ Ожидание первого кадра...</div>',
                        unsafe_allow_html=True)
            else:
                # Поток завершился сам (конец файла)
                if _vs is not None and not _vs.is_alive():
                    new_v = _vs.pop_violations()
                    if new_v:
                        st.session_state.violations.extend(new_v)
                    tmp = getattr(_vs, "_tmp_path", None)
                    if tmp and os.path.exists(tmp):
                        try: os.unlink(tmp)
                        except Exception: pass
                    st.session_state.video_stream = None
                frame_ph.markdown(
                    '<div class="video-info">📷 Нажмите кнопку для начала анализа</div>',
                    unsafe_allow_html=True)

            # Журнал — обновляется внутри фрагмента (не скроллит страницу)
            journal_ph.markdown(journal_html(st.session_state.violations),
                                 unsafe_allow_html=True)

            # Кнопка скачивания — только когда не запущено, внутри фрагмента
            _violations = st.session_state.violations
            if _violations and not _run:
                _counts = defaultdict(int)
                for v in _violations:
                    _counts[v["type"]] += 1
                _report = {
                    "date":       datetime.now().strftime("%Y-%m-%d"),
                    "total":      len(_violations),
                    "counts":     dict(_counts),
                    "violations": _violations,
                }
                st.download_button(
                    key="dl_report",
                    label="⬇️ Скачать JSON-отчёт",
                    data=json.dumps(_report, ensure_ascii=False, indent=2),
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    width='stretch',
                )

        _live_view()


if __name__ == "__main__":
    main()