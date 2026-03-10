"""
Часть 1, Этап 3 — Модуль распознавания лиц
Использует face_recognition (dlib) для точного распознавания.
"""

import cv2
import os
import pickle
import numpy as np

try:
    import face_recognition as fr
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False

FACES_DB_PATH = "database/faces_db.pkl"
FACES_DIR = "outputs/faces"
FRAME_STEP = 10
CASCADE_PATH = "haarcascade_frontalface_default.xml"

RECOGNITION_THRESHOLD = 0.55  # чем меньше — тем строже (расстояние между векторами)


def load_faces_db():
    if not os.path.exists(FACES_DB_PATH):
        return {}
    with open(FACES_DB_PATH, "rb") as f:
        return pickle.load(f)


def save_faces_db(db):
    os.makedirs(os.path.dirname(FACES_DB_PATH), exist_ok=True)
    with open(FACES_DB_PATH, "wb") as f:
        pickle.dump(db, f)


def get_encoding_from_frame(frame):
    """Возвращает (encoding, face_crop) или (None, None)."""
    if not FR_AVAILABLE:
        return None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model="hog")
    if not locations:
        return None, None
    encodings = fr.face_encodings(rgb, locations)
    if not encodings:
        return None, None
    # Берём самое большое лицо
    best_idx = max(range(len(locations)),
                   key=lambda i: (locations[i][2]-locations[i][0]) * (locations[i][1]-locations[i][3]))
    top, right, bottom, left = locations[best_idx]
    face_crop = frame[top:bottom, left:right]
    return encodings[best_idx], face_crop


def get_all_face_encodings(frame):
    """Возвращает список (encoding, box) для всех лиц в кадре."""
    if not FR_AVAILABLE:
        return []
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(rgb, model="hog")
        if not locations:
            return []
        encodings = fr.face_encodings(rgb, locations)
        result = []
        for i, (top, right, bottom, left) in enumerate(locations):
            if i < len(encodings):
                result.append({
                    "encoding": encodings[i],
                    "box": [left, top, right, bottom],  # x1,y1,x2,y2
                })
        return result
    except Exception:
        return []


def recognize_from_encoding(encoding, db):
    """Сравнивает encoding со всеми в базе. Возвращает (имя, дистанция) или (None, dist)."""
    if encoding is None or not db:
        return None, 1.0

    known_names = []
    known_encodings = []
    for name, info in db.items():
        enc = info.get("encoding") if isinstance(info, dict) else None
        if enc is not None:
            known_names.append(name)
            known_encodings.append(enc)

    if not known_encodings:
        return None, 1.0

    distances = fr.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= RECOGNITION_THRESHOLD:
        return known_names[best_idx], best_dist
    return None, best_dist


def detect_and_save_face(frame, violation_id):
    """Ищет лицо на кадре, сохраняет и возвращает (путь, encoding)."""
    encoding, face_crop = get_encoding_from_frame(frame)

    if face_crop is None or face_crop.size == 0:
        # Fallback: Haar-каскад
        cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if not cascade.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                face_crop = frame[y:y+h, x:x+w]

    if face_crop is None or face_crop.size == 0:
        return None, None

    os.makedirs(FACES_DIR, exist_ok=True)
    face_path = os.path.join(FACES_DIR, f"face_{violation_id}.jpg")
    cv2.imwrite(face_path, face_crop)
    return face_path, encoding


def extract_best_face_from_video(video_path, frame_step=FRAME_STEP, target_box=None):
    """Анализирует видеосегмент, возвращает лицо наиболее близкое к target_box.
    target_box = [x1,y1,x2,y2] — координаты нарушителя из MediaPipe.
    Если target_box не задан — берёт самое большое лицо из первой половины видео."""
    if not os.path.exists(video_path):
        return None, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Берём первую половину — там лицо ещё с открытыми глазами
    end_frame = max(1, int(total_frames * 0.5))

    best_face = None
    best_encoding = None
    best_score = -1
    frame_idx = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0 and FR_AVAILABLE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb, model="hog")
            if locations:
                encs = fr.face_encodings(rgb, locations)
                fh, fw = frame.shape[:2]
                for i, (top, right, bottom, left) in enumerate(locations):
                    if i >= len(encs):
                        continue
                    area = (bottom - top) * (right - left)
                    if target_box is not None:
                        # Нормализуем box к размеру кадра
                        tx1 = target_box[0] / fw
                        ty1 = target_box[1] / fh
                        tx2 = target_box[2] / fw
                        ty2 = target_box[3] / fh
                        lx1 = left / fw; ly1 = top / fh
                        lx2 = right / fw; ly2 = bottom / fh
                        # Расстояние между центрами (меньше = лучше)
                        cx_t = (tx1 + tx2) / 2; cy_t = (ty1 + ty2) / 2
                        cx_l = (lx1 + lx2) / 2; cy_l = (ly1 + ly2) / 2
                        dist = ((cx_t - cx_l)**2 + (cy_t - cy_l)**2) ** 0.5
                        score = 1.0 - dist  # больше = ближе
                    else:
                        score = area  # без target — самое большое
                    if score > best_score:
                        best_score = score
                        best_face = frame[top:bottom, left:right].copy()
                        best_encoding = encs[i]
        frame_idx += 1

    cap.release()
    return best_face, best_encoding


def process_incidents_faces(incidents):
    """Ищет лица в видеосегментах и распознаёт студентов."""
    db = load_faces_db()
    print(f"\n[INFO] Поиск лиц в {len(incidents)} инцидентах (face_recognition)...")
    os.makedirs(FACES_DIR, exist_ok=True)

    for inc in incidents:
        face_img, encoding = None, None

        if inc.get("video_path") and os.path.exists(inc["video_path"]):
            target_box = inc.get("face_box")  # координаты спящего лица из MediaPipe
            face_img, encoding = extract_best_face_from_video(inc["video_path"], target_box=target_box)

        if face_img is None and inc.get("face_path") and os.path.exists(inc["face_path"]):
            img = cv2.imread(inc["face_path"])
            if img is not None:
                encoding, face_img = get_encoding_from_frame(img)

        if face_img is not None and face_img.size > 0:
            face_path = os.path.join(FACES_DIR, f"face_id{inc['id']}.jpg")
            cv2.imwrite(face_path, face_img)
            inc["face_path"] = face_path

            if db and encoding is not None:
                name, dist = recognize_from_encoding(encoding, db)
                rt_name = inc.get("student_name")
                # Для сна — доверяем видео (глаза открыты в начале сегмента)
                # Для остальных — real-time надёжнее
                if rt_name and inc.get("type") != "sleeping":
                    inc["student_name"] = rt_name
                    inc["recognition_confidence"] = 100.0
                    print(f"  [ID {inc['id']}] → {rt_name} (real-time)")
                else:
                    inc["student_name"] = name if name else "Неизвестный"
                    inc["recognition_confidence"] = round((1 - dist) * 100, 1)
                    print(f"  [ID {inc['id']}] → {inc['student_name']} (дистанция: {dist:.3f})")
            else:
                rt_name = inc.get("student_name")
                if rt_name and inc.get("type") != "sleeping":
                    inc["student_name"] = rt_name
                else:
                    inc["student_name"] = "Неизвестный"
                inc["recognition_confidence"] = 0
                print(f"  [ID {inc['id']}] → {inc['student_name']}")
        else:
            inc["student_name"] = "Лицо не найдено"
            inc["recognition_confidence"] = 0
            print(f"  [ID {inc['id']}] → Лицо не найдено")

    return incidents


# Оставляем для совместимости со старым кодом
def face_to_histogram(face_img):
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    resized = cv2.resize(gray, (100, 100))
    hist = cv2.calcHist([resized], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def recognize_face(face_hist, db, threshold=0.7):
    """Совместимость со старым кодом — используется в draw_face_label."""
    return None, 0.0


if __name__ == "__main__":
    students = list(load_faces_db().keys())
    print(f"Студенты в базе: {', '.join(students)}" if students else "База данных пуста.")