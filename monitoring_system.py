"""
Часть 1, Этап 2 — Основной модуль мониторинга дисциплины
Использует YOLO для детекции нарушений + MediaPipe для детекции сна по закрытым глазам.
"""

import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO
from fr_module import detect_and_save_face, load_faces_db, recognize_from_encoding, get_encoding_from_frame, get_all_face_encodings

# MediaPipe для детекции закрытых глаз
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[WARNING] mediapipe не установлен. Детекция сна недоступна.")
    print("          Установите: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# ─────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────

MODEL_PATH = "models/violations_detector.pt"

CONFIDENCE_THRESHOLD = 0.5
VIOLATION_BUFFER_SEC = 5
MIN_VIOLATION_SEC = 2
FRAME_SKIP = 2

# Глаза считаются закрытыми если EAR < порога
EAR_THRESHOLD = 0.22
# Сон регистрируется если глаза закрыты дольше N секунд
SLEEP_TRIGGER_SEC = 5.0

OUTPUT_DIR = "outputs"
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")
FACES_DIR = os.path.join(OUTPUT_DIR, "faces")

VIOLATION_CLASSES = {
    "phone_usage": [67],
    "bottle":      [39],
    "food":        [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],
    "sleeping":    ["sleeping"],
}

VIOLATION_NAMES = {
    "phone_usage": "Phone usage",
    "bottle":      "Bottle/drink",
    "food":        "Food",
    "sleeping":    "Sleeping",
}

VIOLATION_COLORS = {
    "phone_usage": (0, 0, 255),
    "bottle":      (0, 165, 255),
    "food":        (0, 255, 255),
    "sleeping":    (255, 0, 0),
}


# ─────────────────────────────────────────────
# ДЕТЕКТОР СНА (MediaPipe)
# ─────────────────────────────────────────────

# Индексы точек глаз в сетке MediaPipe Face Mesh (468 точек)
# Левый глаз: верх/низ/лево/право
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    """
    Вычисляет EAR (Eye Aspect Ratio) — отношение высоты глаза к ширине.
    Чем меньше EAR — тем более закрыт глаз.
    Формула: EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    # Вертикальные расстояния
    v1 = ((pts[1][0]-pts[5][0])**2 + (pts[1][1]-pts[5][1])**2) ** 0.5
    v2 = ((pts[2][0]-pts[4][0])**2 + (pts[2][1]-pts[4][1])**2) ** 0.5
    # Горизонтальное расстояние
    h  = ((pts[0][0]-pts[3][0])**2 + (pts[0][1]-pts[3][1])**2) ** 0.5

    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


class SleepDetector:
    """
    Детектор сна на основе MediaPipe Face Mesh.
    Отслеживает EAR обоих глаз и фиксирует нарушение
    если глаза закрыты дольше SLEEP_TRIGGER_SEC секунд.
    """

    def __init__(self, ear_threshold=EAR_THRESHOLD, trigger_sec=SLEEP_TRIGGER_SEC):
        self.ear_threshold = ear_threshold
        self.trigger_sec = trigger_sec
        self._closed_since = {}   # {face_key: время закрытия глаз}
        self._closed_frame = {}   # {face_key: кадр в момент закрытия (для фото)}
        self._face_mesh = None

    def _get_mesh(self):
        if self._face_mesh is None and MEDIAPIPE_AVAILABLE:
            try:
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks import python as mp_python
                # Читаем файл как байты — обходим проблему кириллицы в пути
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
                with open(model_path, "rb") as f:
                    model_data = f.read()
                base_opts = mp_python.BaseOptions(model_asset_buffer=model_data)
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    num_faces=5,
                    min_face_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._face_mesh = mp_vision.FaceLandmarker.create_from_options(opts)
                self._is_tasks_api = True
                print("[INFO] MediaPipe Tasks API инициализирован.")
            except Exception as e:
                print(f"[ERROR] MediaPipe не инициализирован: {e}")
                self._face_mesh = None
        return self._face_mesh

    def _face_key(self, box, w, h):
        """Уникальный ключ лица по позиции в сетке 10x10."""
        cx = int((box[0] + box[2]) / 2 / w * 10)
        cy = int((box[1] + box[3]) / 2 / h * 10)
        return (cx, cy)

    def process(self, frame, current_time):
        """
        Анализирует кадр. Возвращает список словарей — по одному на каждое лицо:
          [{"sleeping": bool, "eyes_closed": bool, "closed_duration": float,
            "ear": float, "face_box": [x1,y1,x2,y2]}, ...]
        """
        mesh = self._get_mesh()
        if mesh is None:
            return []

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Получаем список landmark-ов всех лиц
        all_face_lms = []
        if getattr(self, '_is_tasks_api', False):
            import mediapipe as mp2
            mp_img = mp2.Image(image_format=mp2.ImageFormat.SRGB, data=rgb_frame)
            mesh_result = mesh.detect(mp_img)
            all_face_lms = mesh_result.face_landmarks if mesh_result.face_landmarks else []
        else:
            mesh_result = mesh.process(rgb_frame)
            if mesh_result.multi_face_landmarks:
                all_face_lms = [fl.landmark for fl in mesh_result.multi_face_landmarks]

        # Убираем лица которые пропали из кадра
        active_keys = set()
        results = []

        for lms in all_face_lms:
            xs = [lm.x * w for lm in lms]
            ys = [lm.y * h for lm in lms]
            face_box = [min(xs), min(ys), max(xs), max(ys)]
            key = self._face_key(face_box, w, h)
            active_keys.add(key)

            left_ear  = eye_aspect_ratio(lms, LEFT_EYE,  w, h)
            right_ear = eye_aspect_ratio(lms, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            eyes_closed = avg_ear < self.ear_threshold

            # Таймер для каждого лица отдельно
            if eyes_closed:
                if key not in self._closed_since:
                    self._closed_since[key] = current_time
                    self._closed_frame[key] = frame.copy()  # сохраняем кадр пока глаза ещё видны
                closed_duration = current_time - self._closed_since[key]
            else:
                self._closed_since.pop(key, None)
                self._closed_frame.pop(key, None)
                closed_duration = 0.0

            sleeping = eyes_closed and closed_duration >= self.trigger_sec

            results.append({
                "sleeping": sleeping,
                "eyes_closed": eyes_closed,
                "closed_duration": closed_duration,
                "ear": avg_ear,
                "face_box": face_box,
                "face_key": key,
                "landmarks": lms,
                "closed_frame": self._closed_frame.get(key),  # кадр с открытыми глазами
            })

        # Убираем таймеры пропавших лиц
        for key in list(self._closed_since.keys()):
            if key not in active_keys:
                del self._closed_since[key]

        return results


# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def ensure_dirs():
    for d in [SEGMENTS_DIR, FACES_DIR]:
        os.makedirs(d, exist_ok=True)


def get_violation_type(class_id, class_name):
    for vtype, ids in VIOLATION_CLASSES.items():
        if class_id in ids or class_name in ids:
            return vtype
    return None


def draw_annotation(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame,
                  (x1, y1 - text_size[1] - 8),
                  (x1 + text_size[0] + 4, y1),
                  color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def draw_sleep_overlay(frame, sleep_results):
    """Рисует рамку вокруг глаз и статус для каждого лица."""
    h, w = frame.shape[:2]
    any_sleeping = False

    for info in sleep_results:
        closed = info["eyes_closed"]
        dur = info["closed_duration"]
        sleeping = info["sleeping"]
        lms = info.get("landmarks")
        if sleeping:
            any_sleeping = True

        if lms is not None:
            # Вычисляем bounding box только по точкам глаз
            eye_pts = LEFT_EYE + RIGHT_EYE
            xs = [lms[i].x * w for i in eye_pts]
            ys = [lms[i].y * h for i in eye_pts]
            pad = int((max(xs) - min(xs)) * 0.4)  # отступ 40% от ширины глаз
            ex1 = max(0,    int(min(xs)) - pad)
            ey1 = max(0,    int(min(ys)) - pad)
            ex2 = min(w-1,  int(max(xs)) + pad)
            ey2 = min(h-1,  int(max(ys)) + pad)

            color = (0, 0, 255) if sleeping else (0, 140, 255) if closed else (0, 200, 0)
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), color, 2)

            if closed:
                label = f"Sleep! {dur:.0f}s" if sleeping else f"Closed {dur:.1f}s/{SLEEP_TRIGGER_SEC:.0f}s"
                cv2.putText(frame, label, (ex1, ey1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if any_sleeping:
        cv2.putText(frame, "! SLEEPING !", (w//2 - 160, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    return frame


def transliterate(text):
    """Транслитерация кириллицы в латиницу для отображения на видео."""
    table = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh',
        'з':'z','и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o',
        'п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts',
        'ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu',
        'я':'ya',
        'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ё':'Yo','Ж':'Zh',
        'З':'Z','И':'I','Й':'Y','К':'K','Л':'L','М':'M','Н':'N','О':'O',
        'П':'P','Р':'R','С':'S','Т':'T','У':'U','Ф':'F','Х':'Kh','Ц':'Ts',
        'Ч':'Ch','Ш':'Sh','Щ':'Sch','Ъ':'','Ы':'Y','Ь':'','Э':'E','Ю':'Yu',
        'Я':'Ya',
    }
    return ''.join(table.get(c, c) for c in text)


def draw_face_label(frame, db, cached_face_data=None):
    """
    Рисует рамку и имя для каждого лица используя кэшированные данные.
    Если кэш пуст — пропускаем (не вызываем dlib повторно).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    faces = cached_face_data or []
    for fd in faces:
        x1, y1, x2, y2 = map(int, fd["box"])
        raw_name = fd.get("name")
        name = transliterate(raw_name) if raw_name else "Unknown"
        color = (0, 200, 0) if raw_name else (160, 160, 160)

        if raw_name:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        cv2.rectangle(frame, (x1, y2), (x1 + tw + 6, y2 + th + 8), color, -1)
        cv2.putText(frame, name, (x1 + 3, y2 + th + 4),
                    font, font_scale, (255, 255, 255), thickness)

    return frame


# ─────────────────────────────────────────────
# КЛАСС ТРЕКЕРА НАРУШЕНИЙ
# ─────────────────────────────────────────────

def _iou(box1, box2):
    """Пересечение/объединение двух боксов для сопоставления нарушений."""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class ViolationTracker:
    """
    Трекер нарушений по позиции бокса (IoU).
    Поддерживает несколько одновременных нарушений одного типа.
    Ключ: (тип, track_id) — уникален для каждого нарушителя.
    """
    IOU_THRESHOLD = 0.3   # минимальное совпадение боксов для сопоставления

    def __init__(self, buffer_sec=VIOLATION_BUFFER_SEC, min_duration=MIN_VIOLATION_SEC):
        self.buffer_sec = buffer_sec
        self.min_duration = min_duration
        self.active = {}   # {track_key: info_dict}
        self.incidents = []
        self._id_counter = 0
        self._track_counter = 0

    def _next_id(self):
        self._id_counter += 1
        return self._id_counter

    def _next_track(self):
        self._track_counter += 1
        return self._track_counter

    def _match(self, vtype, box):
        """Ищет активное нарушение того же типа с похожим боксом.
        Для сна используем мягкий IoU порог — бокс лица немного плавает между кадрами."""
        best_key = None
        best_iou = 0.0
        # Для сна используем низкий порог IoU (бокс лица слегка плавает)
        threshold = 0.05 if vtype == "sleeping" else self.IOU_THRESHOLD
        for key, info in self.active.items():
            if info["type"] != vtype:
                continue
            iou = _iou(box, info["box"])
            if iou > best_iou:
                best_iou = iou
                best_key = key
        if best_iou >= threshold:
            return best_key
        return None

    def update(self, detections, current_time, frame=None, face_data=None):
        """
        face_data: список {"encoding": ..., "box": [x1,y1,x2,y2], "name": str} — все лица в кадре
        """
        matched_keys = set()

        for d in detections:
            vtype = d["type"]
            box = d.get("box", [0, 0, 100, 100])
            conf = d.get("conf", 1.0)

            key = self._match(vtype, box)
            if key is None:
                # Новое нарушение
                vid = self._next_id()
                track = self._next_track()
                key = (vtype, track)
                self.active[key] = {
                    "id": vid,
                    "type": vtype,
                    "start": current_time,
                    "last_seen": current_time,
                    "box": box,
                    "conf": conf,
                    "face_path": None,
                    "student_name": None,
                    "best_encoding": None,
                    "face_box": d.get("face_box"),  # для сна — bbox MediaPipe
                }
                # Для не-сна сохраняем фото сразу
                # Для сна — фото будет извлечено из видеосегмента в Этапе 3
                if frame is not None and vtype != "sleeping":
                    face_path, _ = detect_and_save_face(frame, vid)
                    self.active[key]["face_path"] = face_path

            else:
                self.active[key]["last_seen"] = current_time
                self.active[key]["box"] = box

            # Привязываем ближайшее лицо к нарушению
            if face_data:
                vbox = self.active[key]["box"]
                best_iou = 0.0
                best_face = None
                for fd in face_data:
                    iou = _iou(vbox, fd["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_face = fd
                if best_face and best_iou > 0.01:
                    # Привязываем имя если распознали
                    if best_face.get("name"):
                        self.active[key]["student_name"] = best_face["name"]

            matched_keys.add(key)

        finished = []
        for key in list(self.active.keys()):
            info = self.active[key]
            vtype = info["type"]
            time_since_last = current_time - info["last_seen"]
            # Для сна: нет буфера и нет минимальной длительности — уже подтверждено MediaPipe
            buf = 1.0 if vtype == "sleeping" else self.buffer_sec
            if key not in matched_keys and time_since_last > buf:
                duration = info["last_seen"] - info["start"]
                min_dur = 0 if vtype == "sleeping" else self.min_duration
                if duration >= min_dur:
                    incident = {
                        "id": info["id"],
                        "type": info["type"],
                        "start_time": info["start"],
                        "end_time": info["last_seen"],
                        "duration": duration,
                        "face_path": info["face_path"],
                        "face_box": info.get("face_box"),
                        "student_name": info.get("student_name"),
                        "video_path": None,
                    }
                    self.incidents.append(incident)
                    finished.append(incident)
                del self.active[key]

        return list(self.active.values()), finished

    def get_active_list(self):
        return list(self.active.values())

    def force_close_all(self, current_time):
        for key, info in list(self.active.items()):
            duration = info["last_seen"] - info["start"]
            vtype = info["type"]
            # Для сна минимальная длительность = SLEEP_TRIGGER_SEC (уже подтверждено)
            # Для остальных — MIN_VIOLATION_SEC
            min_dur = 0 if vtype == "sleeping" else self.min_duration
            if duration >= min_dur:
                self.incidents.append({
                    "id": info["id"],
                    "type": vtype,
                    "start_time": info["start"],
                    "end_time": current_time,
                    "duration": current_time - info["start"],
                    "face_path": info["face_path"],
                    "student_name": info.get("student_name"),
                    "video_path": None,
                })
        self.active.clear()


# ─────────────────────────────────────────────
# КЛАСС ЗАПИСИ ВИДЕОСЕГМЕНТОВ
# ─────────────────────────────────────────────

class SegmentRecorder:
    def __init__(self, fps=20, buffer_before_sec=3, buffer_after_sec=VIOLATION_BUFFER_SEC):
        self.fps = fps
        self.buffer_before = int(buffer_before_sec * fps)
        self.buffer_after_sec = buffer_after_sec
        self.pre_buffer = []
        self.writers = {}

    def add_frame_to_prebuffer(self, frame):
        self.pre_buffer.append(frame.copy())
        if len(self.pre_buffer) > self.buffer_before:
            self.pre_buffer.pop(0)

    def start_recording(self, violation_id, vtype, frame_size):
        if violation_id in self.writers:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SEGMENTS_DIR, f"violation_{vtype}_{ts}_id{violation_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, frame_size)
        for f in self.pre_buffer:
            writer.write(f)
        self.writers[violation_id] = {
            "writer": writer,
            "path": path,
            "last_active": time.time(),
        }
        return path

    def write_frame(self, violation_id, frame):
        if violation_id in self.writers:
            self.writers[violation_id]["writer"].write(frame)
            self.writers[violation_id]["last_active"] = time.time()

    def stop_recording(self, violation_id):
        if violation_id in self.writers:
            info = self.writers.pop(violation_id)
            info["writer"].release()
            return info["path"]
        return None

    def stop_all(self):
        for vid in list(self.writers.keys()):
            self.stop_recording(vid)


# ─────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ МОНИТОРИНГА
# ─────────────────────────────────────────────

def run_monitoring(source=0, show_window=True, save_segments=True):
    ensure_dirs()

    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else "yolo11n.pt"
    print(f"[INFO] Загружаем модель: {model_path}")
    model = YOLO(model_path)
    class_names = model.names

    if MEDIAPIPE_AVAILABLE:
        print(f"[INFO] Детекция сна активна (порог EAR={EAR_THRESHOLD}, триггер={SLEEP_TRIGGER_SEC}с)")
        sleep_detector = SleepDetector()
    else:
        sleep_detector = None

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть источник видео: {source}")
        return [], None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_w, frame_h)

    tracker = ViolationTracker()
    recorder = SegmentRecorder(fps=fps) if save_segments else None

    # Загружаем базу лиц для отображения имён в реальном времени
    _faces_db = load_faces_db()
    if _faces_db:
        print(f"[INFO] База лиц загружена: {len(_faces_db)} студентов")
    else:
        print("[INFO] База лиц пуста — лица будут подписаны 'Unknown'")

    frame_count = 0
    _cached_face_data = []  # кэш распознавания лиц (обновляется раз в 15 кадров)
    start_ts = datetime.now()
    print(f"[INFO] Мониторинг начат: {start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[INFO] Нажмите 'q' для остановки.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Видео завершено или поток прерван.")
                break

            frame_count += 1
            current_time = time.time()

            if recorder:
                recorder.add_frame_to_prebuffer(frame)

            if frame_count % FRAME_SKIP != 0:
                if show_window:
                    cv2.imshow("Discipline Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            annotated_frame = frame.copy()
            detections = []

            # ── Детекция YOLO (телефон, бутылка, еда) ──
            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names.get(cls_id, str(cls_id))
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    vtype = get_violation_type(cls_id, cls_name)
                    if vtype is None or vtype == "sleeping":
                        continue  # сон детектируем через MediaPipe

                    detections.append({"type": vtype, "box": xyxy, "conf": conf})
                    color = VIOLATION_COLORS.get(vtype, (0, 255, 0))
                    label = f"{VIOLATION_NAMES.get(vtype, vtype)} {conf:.0%}"
                    draw_annotation(annotated_frame, xyxy, label, color)

            # ── Детекция сна через MediaPipe (все лица) ──
            if sleep_detector:
                sleep_results = sleep_detector.process(frame, current_time)
                annotated_frame = draw_sleep_overlay(annotated_frame, sleep_results)

                for sinfo in sleep_results:
                    if sinfo["sleeping"]:
                        box = sinfo["face_box"] or [0, 0, frame_w, frame_h]
                        detections.append({
                            "type": "sleeping",
                            "box": box,
                            "conf": 1.0,
                            "face_box": box,
                            "closed_frame": sinfo.get("closed_frame"),  # кадр с открытыми глазами
                        })

            # ── Распознавание лиц (раз в 15 кадров, результат кэшируется) ──
            if frame_count % 5 == 0:
                _cached_face_data = []
                for fd in get_all_face_encodings(frame):
                    name, dist = recognize_from_encoding(fd["encoding"], _faces_db)
                    fd["name"] = name
                    fd["dist"] = dist
                    _cached_face_data.append(fd)

            # ── Обновление трекера (используем кэш) ──
            active, finished = tracker.update(detections, current_time, frame, face_data=_cached_face_data)

            # ── Управление записью ──
            if recorder:
                for a in active:
                    vid = a["id"]
                    if vid not in recorder.writers:
                        recorder.start_recording(vid, a.get("type", "unknown"), frame_size)
                    recorder.write_frame(vid, annotated_frame)

                for incident in finished:
                    path = recorder.stop_recording(incident["id"])
                    if path:
                        incident["video_path"] = path
                        print(f"  [SAVED] Сегмент: {path}")

            # ── Рамки и имена лиц на кадре ──
            draw_face_label(annotated_frame, _faces_db, cached_face_data=_cached_face_data)

            # ── Статистика на кадре ──
            y_offset = 30
            cv2.putText(annotated_frame,
                        f"Active: {len(active)} | Total: {len(tracker.incidents)}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            for a in active:
                y_offset += 25
                duration = current_time - a["start"]
                name = VIOLATION_NAMES.get(a.get("type"), "?")
                cv2.putText(annotated_frame,
                            f"  {name}: {duration:.0f}s",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if show_window:
                cv2.imshow("Discipline Monitor", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        now = time.time()
        tracker.force_close_all(now)
        if recorder:
            # Сохраняем видео для инцидентов которые были активны при завершении
            for incident in tracker.incidents:
                if incident.get("video_path") is None:
                    path = recorder.stop_recording(incident["id"])
                    if path:
                        incident["video_path"] = path
                        print(f"  [SAVED] Сегмент: {path}")
            recorder.stop_all()
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    end_ts = datetime.now()
    print(f"\n[INFO] Мониторинг завершён: {end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Зафиксировано инцидентов: {len(tracker.incidents)}")

    return tracker.incidents, start_ts, end_ts


if __name__ == "__main__":
    incidents, start, end = run_monitoring(source=0)
    for inc in incidents:
        print(f"  #{inc['id']} | {VIOLATION_NAMES.get(inc['type'], inc['type'])} | "
              f"{inc['duration']:.1f}с | видео: {inc.get('video_path')}")
