"""
Часть 1 — ГЛАВНЫЙ ФАЙЛ
Запускает полный цикл системы мониторинга:
  1. Мониторинг видеопотока (YOLO-детекция + запись)
  2. Распознавание лиц из сохранённых сегментов
  3. Генерация отчёта
"""

from monitoring_system import run_monitoring
from fr_module import process_incidents_faces
from report_generator import generate_text_report, save_incidents_json


def main(source=0, show_window=True):
    """
    Параметры:
        source      — 0 (веб-камера), путь к файлу или URL потока
        show_window — показывать окно OpenCV во время мониторинга
    """
    print("=" * 60)
    print("  СИСТЕМА МОНИТОРИНГА ДИСЦИПЛИНЫ НА ЗАНЯТИЯХ")
    print("=" * 60)

    # ── Этап 2: Основной мониторинг ──
    print("\n[ЭТАП 2] Запуск мониторинга видеопотока...")
    incidents, start_time, end_time = run_monitoring(
        source=source,
        show_window=show_window,
        save_segments=True,
    )

    if not incidents:
        print("\nНарушений не зафиксировано.")
        generate_text_report([], start_time, end_time)
        return

    # ── Этап 3: Распознавание лиц ──
    print(f"\n[ЭТАП 3] Распознавание лиц в {len(incidents)} инцидентах...")
    incidents = process_incidents_faces(incidents)

    # ── Этап 4: Отчёт ──
    print("\n[ЭТАП 4] Формирование отчёта...")
    generate_text_report(incidents, start_time, end_time, save_to_file=True)
    save_incidents_json(incidents, start_time, end_time)

    print("\n[ГОТОВО] Все файлы сохранены в директории 'outputs/'")


if __name__ == "__main__":
    import sys

    # Можно передать источник как аргумент командной строки
    # python main.py                    → веб-камера
    # python main.py video.mp4          → видеофайл
    # python main.py rtsp://...         → IP-камера

    source = sys.argv[1] if len(sys.argv) > 1 else 0
    # Если аргумент — цифра, преобразуем в int (индекс камеры)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    main(source=source, show_window=True)
