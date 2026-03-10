"""
Часть 1, Этап 4 — Формирование отчёта о нарушениях дисциплины
"""

import os
import json
from datetime import datetime


# ─────────────────────────────────────────────
# ЧЕЛОВЕКОЧИТАЕМЫЕ НАЗВАНИЯ
# ─────────────────────────────────────────────

VIOLATION_NAMES = {
    "phone_usage": "Использование телефона",
    "bottle":      "Бутылка/напиток",
    "food":        "Еда на занятии",
    "sleeping":    "Сон на занятии",
}

OUTPUT_DIR = os.path.join("outputs", "reports")


# ─────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────

def fmt_time(ts):
    """Форматирует UNIX timestamp в строку ЧЧ:ММ:СС."""
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_duration(seconds):
    """Форматирует длительность в секундах."""
    sec = int(seconds)
    if sec < 60:
        return f"{sec} сек"
    return f"{sec // 60} мин {sec % 60} сек"


# ─────────────────────────────────────────────
# ГЕНЕРАЦИЯ ТЕКСТОВОГО ОТЧЁТА
# ─────────────────────────────────────────────

def generate_text_report(incidents, start_time, end_time, save_to_file=True):
    """
    Генерирует текстовый отчёт о нарушениях в стиле, указанном в задании.

    Параметры:
        incidents   — список инцидентов (из ViolationTracker)
        start_time  — datetime начала мониторинга
        end_time    — datetime конца мониторинга
        save_to_file — сохранить в файл outputs/report_ДАТА.txt

    Возвращает: строку с текстом отчёта
    """
    lines = []
    sep_long  = "═" * 72
    sep_short = "─" * 72

    # ── Заголовок ──
    lines.append(sep_long)
    lines.append("                    ОТЧЁТ О НАРУШЕНИЯХ ДИСЦИПЛИНЫ")
    lines.append(f"                    Дата: {start_time.strftime('%Y-%m-%d')}")
    lines.append(
        f"                    Время мониторинга: "
        f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}"
    )
    total_duration = (end_time - start_time).seconds
    lines.append(
        f"                    Продолжительность: {fmt_duration(total_duration)}"
    )
    lines.append(sep_long)
    lines.append("")

    if not incidents:
        lines.append("  Нарушений не зафиксировано.")
        lines.append("")
        lines.append(sep_long)
    else:
        # ── Каждый инцидент ──
        for idx, inc in enumerate(incidents, start=1):
            lines.append(f"№{idx}. НАРУШЕНИЕ")
            lines.append(sep_short)

            t_start = fmt_time(inc["start_time"])
            t_end   = fmt_time(inc["end_time"])
            dur     = fmt_duration(inc["duration"])
            lines.append(f"  Время:        {t_start} - {t_end} ({dur})")

            vname = VIOLATION_NAMES.get(inc["type"], inc["type"])
            lines.append(f"  Тип:          {vname} ({inc['type']})")

            student = inc.get("student_name") or "Неизвестный"
            conf    = inc.get("recognition_confidence", 0.0)
            if student not in ("Неизвестный", "Лицо не найдено", None):
                lines.append(f"  Нарушитель:   {student} (уверенность: {conf:.1f}%)")
            else:
                lines.append(f"  Нарушитель:   {student}")

            if inc.get("video_path"):
                lines.append(f"  Видеозапись:  {inc['video_path']}")
            else:
                lines.append("  Видеозапись:  не сохранена")

            if inc.get("face_path"):
                lines.append(f"  Фото лица:    {inc['face_path']}")

            lines.append("")

        # ── Статистика ──
        lines.append(sep_long)
        lines.append("  ИТОГОВАЯ СТАТИСТИКА")
        lines.append(sep_short)
        lines.append(f"  Всего нарушений: {len(incidents)}")

        # Группировка по типу
        by_type = {}
        for inc in incidents:
            by_type[inc["type"]] = by_type.get(inc["type"], 0) + 1
        for vtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"    • {VIOLATION_NAMES.get(vtype, vtype)}: {count}")

        # Уникальные нарушители
        students = [
            inc.get("student_name")
            for inc in incidents
            if inc.get("student_name") and
               inc.get("student_name") not in ("Неизвестный", "Лицо не найдено")
        ]
        if students:
            lines.append(f"\n  Идентифицировано нарушителей: {len(set(students))}")
            for s in sorted(set(students)):
                cnt = students.count(s)
                lines.append(f"    • {s}: {cnt} нарушений")

        lines.append(sep_long)

    report_text = "\n".join(lines)

    # ── Вывод в консоль ──
    print("\n" + report_text)

    # ── Сохранение в файл ──
    if save_to_file:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = f"report_{start_time.strftime('%Y%m%d_%H%M%S')}.txt"
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[INFO] Отчёт сохранён: {path}")

    return report_text


# ─────────────────────────────────────────────
# СОХРАНЕНИЕ В JSON (для Streamlit / внешних систем)
# ─────────────────────────────────────────────

def save_incidents_json(incidents, start_time, end_time):
    """
    Сохраняет инциденты в JSON для дальнейшей обработки или веб-интерфейса.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"incidents_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(OUTPUT_DIR, filename)

    data = {
        "session": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_sec": (end_time - start_time).seconds,
        },
        "incidents": [
            {
                "id": inc["id"],
                "type": inc["type"],
                "type_name": VIOLATION_NAMES.get(inc["type"], inc["type"]),
                "start_time": fmt_time(inc["start_time"]),
                "end_time": fmt_time(inc["end_time"]),
                "duration_sec": round(inc["duration"], 1),
                "student_name": inc.get("student_name") or "Неизвестный",
                "recognition_confidence": round(inc.get("recognition_confidence", 0.0), 2),
                "video_path": inc.get("video_path"),
                "face_path": inc.get("face_path"),
            }
            for inc in incidents
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] JSON-отчёт сохранён: {path}")
    return path


# ─────────────────────────────────────────────
# ТОЧКА ВХОДА (для тестирования)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Пример с тестовыми данными
    import time

    now = time.time()
    test_incidents = [
        {
            "id": 1,
            "type": "phone_usage",
            "start_time": now - 90,
            "end_time": now - 8,
            "duration": 82,
            "student_name": "Иван Иванов",
            "recognition_confidence": 0.87,
            "video_path": "outputs/segments/violation_phone_usage_test.mp4",
            "face_path": "outputs/faces/face_1.jpg",
        },
        {
            "id": 2,
            "type": "sleeping",
            "start_time": now - 60,
            "end_time": now - 30,
            "duration": 30,
            "student_name": "Неизвестный",
            "recognition_confidence": 0.21,
            "video_path": None,
            "face_path": None,
        },
        {
            "id": 3,
            "type": "bottle",
            "start_time": now - 20,
            "end_time": now - 5,
            "duration": 15,
            "student_name": "Мария Петрова",
            "recognition_confidence": 0.91,
            "video_path": "outputs/segments/violation_bottle_test.mp4",
            "face_path": "outputs/faces/face_3.jpg",
        },
    ]

    start_dt = datetime.fromtimestamp(now - 5400)  # 1.5 часа назад
    end_dt   = datetime.fromtimestamp(now)

    generate_text_report(test_incidents, start_dt, end_dt)
    save_incidents_json(test_incidents, start_dt, end_dt)