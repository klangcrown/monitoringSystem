"""
Управление базой лиц студентов (face_recognition / dlib).
Запуск: python manage_db.py
"""

import cv2
import os
import pickle
import numpy as np
from datetime import datetime

try:
    import face_recognition as fr
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    print("[WARNING] face_recognition не установлен!")

DB_PATH = "database/faces_db.pkl"
PHOTOS_DIR = "database/photos"


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_db(db):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def get_face_encoding(frame):
    """Возвращает 128-мерный вектор лица или None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = fr.face_locations(rgb, model="hog")
    if not locations:
        return None, None
    encodings = fr.face_encodings(rgb, locations)
    if not encodings:
        return None, None
    # Берём самое большое лицо
    best_idx = 0
    best_area = 0
    for i, (top, right, bottom, left) in enumerate(locations):
        area = (bottom - top) * (right - left)
        if area > best_area:
            best_area = area
            best_idx = i
    top, right, bottom, left = locations[best_idx]
    face_crop = frame[top:bottom, left:right]
    return encodings[best_idx], face_crop


def add_student():
    if not FR_AVAILABLE:
        print("[ERROR] face_recognition не установлен.")
        return

    print("\n── Регистрация студента ──")
    name = input("Введите имя студента (напр. Иван Иванов): ").strip()
    if not name:
        print("[ERROR] Имя не может быть пустым.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть камеру.")
        return

    print("\nСмотрите прямо в камеру.")
    print("Нажмите ПРОБЕЛ для снимка (нужно 5 снимков).")
    print("Нажмите Q для отмены.\n")

    encodings = []
    saved_face = None
    needed = 5

    while len(encodings) < needed:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = fr.face_locations(rgb, model="hog")

        face_found = len(locations) > 0
        for (top, right, bottom, left) in locations:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        done = len(encodings)
        bar = "█" * done + "░" * (needed - done)
        cv2.putText(display, name, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(display, f"[{bar}] {done}/{needed}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        hint = "SPACE = snapshot" if face_found else "No face..."
        color = (0, 255, 0) if face_found else (0, 0, 255)
        cv2.putText(display, hint, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Student Registration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Отменено.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord(' '):
            enc, face_img = get_face_encoding(frame)
            if enc is not None:
                encodings.append(enc)
                if saved_face is None and face_img is not None:
                    saved_face = face_img
                print(f"  ✓ Снимок {len(encodings)}/{needed}")
                white = frame.copy()
                white[:] = (255, 255, 255)
                cv2.imshow("Student Registration", white)
                cv2.waitKey(120)
            else:
                print("  Лицо не найдено, попробуйте снова")

    cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        print("[ERROR] Снимки не сделаны.")
        return

    avg_encoding = np.mean(encodings, axis=0)

    os.makedirs(PHOTOS_DIR, exist_ok=True)
    safe_name = name.replace(" ", "_")
    dest_photo = os.path.join(PHOTOS_DIR, f"{safe_name}.jpg")
    if saved_face is not None:
        cv2.imwrite(dest_photo, saved_face)

    db = load_db()
    db[name] = {
        "name": name,
        "encoding": avg_encoding,
        "photo_path": dest_photo,
        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_db(db)
    print(f"\n✅ Студент '{name}' успешно добавлен в базу!")


def list_students():
    db = load_db()
    if not db:
        print("\n  База пуста — студентов нет.")
        return
    print(f"\n{'─'*50}")
    print(f"  {'Имя':<30} {'Добавлен':<20}")
    print(f"{'─'*50}")
    for name, info in db.items():
        print(f"  {name:<30} {info.get('added','—'):<20}")
    print(f"{'─'*50}")
    print(f"  Всего: {len(db)} студентов")


def delete_student():
    db = load_db()
    if not db:
        print("\n  База пуста.")
        return
    list_students()
    name = input("\nВведите имя студента для удаления: ").strip()
    if name not in db:
        print(f"  Студент '{name}' не найден.")
        return
    confirm = input(f"  Удалить '{name}'? (да/нет): ").strip().lower()
    if confirm in ("да", "y", "yes"):
        del db[name]
        save_db(db)
        print(f"✅ Студент '{name}' удалён.")
    else:
        print("  Отменено.")


def main():
    print("╔══════════════════════════════════════════╗")
    print("║     БАЗА СТУДЕНТОВ — УПРАВЛЕНИЕ          ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        print("\n  1. Показать всех студентов")
        print("  2. Добавить студента (через камеру)")
        print("  3. Удалить студента")
        print("  0. Выход")

        choice = input("\nВыберите: ").strip()

        if choice == "1":
            list_students()
        elif choice == "2":
            add_student()
        elif choice == "3":
            delete_student()
        elif choice == "0":
            print("До свидания!")
            break
        else:
            print("  Неверный выбор.")


if __name__ == "__main__":
    main()