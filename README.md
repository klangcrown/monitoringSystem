# 📹 Система мониторинга дисциплины на занятиях

Веб-приложение для автоматического обнаружения нарушений дисциплины с использованием компьютерного зрения и нейросетевых моделей. Работает на Windows и macOS.

---

## Возможности

- **Детекция нарушений** через YOLO: использование телефона, еда, напитки
- **Детекция сна** через MediaPipe FaceMesh — фиксирует закрытые глаза дольше N секунд (EAR)
- **Распознавание студентов** через face_recognition (dlib):
  - 🟣 Фиолетовая рамка — известный студент из базы
  - ⬛ Чёрная рамка — неизвестный человек
- **Запись видеосегментов** каждого нарушения в `outputs/segments/`
- **Фото нарушителя** (известного и неизвестного) в `outputs/faces/`
- **Автоматические отчёты** JSON + TXT в `outputs/reports/` при остановке
- Поддержка веб-камеры, видеофайлов и URL-потоков (RTSP/HTTP)
- Настраиваемые параметры детекции через боковую панель

---

## Технологии

| Библиотека | Назначение |
|---|---|
| Python 3.11 | Основной язык |
| Streamlit | Веб-интерфейс |
| Ultralytics YOLO | Детекция телефона, еды, бутылки |
| MediaPipe FaceMesh | Детекция сна по EAR |
| face_recognition / dlib | Распознавание лиц студентов |
| OpenCV | Захват и обработка видео |
| Pillow | Отрисовка кириллицы на кадре |

---

## Структура проекта

```
monitoringSystem/
├── app.py                               # Streamlit интерфейс (Часть 2)
├── main.py                              # Точка входа (Часть 1)
├── monitoring_system.py                 # Детекция нарушений (Часть 1)
├── fr_module.py                         # Распознавание лиц (Часть 1)
├── report_generator.py                  # Генерация отчётов (Часть 1)
├── manage_db.py                         # Управление базой студентов
├── haarcascade_frontalface_default.xml  # Каскадный детектор лиц
├── requirements.txt
├── database/
│   ├── faces_db.pkl                     # База лиц студентов
│   └── photos/                          # Фото студентов
└── outputs/
    ├── segments/                        # Видеозаписи нарушений (.mp4)
    ├── faces/                           # Фото нарушителей (.jpg)
    └── reports/                         # Отчёты (.json + .txt)
```

> **Не входят в репозиторий** (добавлены в `.gitignore`): `yolo11n.pt`, `face_landmarker.task`, `faces_db.pkl`, папка `outputs/`, `venv/`

---

## Установка на Windows

### 1. Клонировать репозиторий
```bash
git clone https://github.com/klangcrown/monitoringSystem.git
cd monitoringSystem
```

### 2. Создать виртуальное окружение
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Установить зависимости
```powershell
pip install ultralytics "numpy==1.26.4" opencv-python onnxruntime "mediapipe==0.10.9"
pip install streamlit Pillow

# dlib — готовый wheel для Windows + Python 3.11
pip install https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install face_recognition
```

### 4. Скачать модели
```powershell
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
# yolo11n.pt скачается автоматически при первом запуске
```

### 5. Скопировать модели dlib (если кириллица в пути)
```powershell
xcopy "venv\Lib\site-packages\face_recognition_models\models" C:\models /E /I
```

---

## Установка на macOS

### 1. Установить Homebrew и Python 3.11
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.11
```

### 2. Клонировать репозиторий
```bash
git clone https://github.com/klangcrown/monitoringSystem.git
cd monitoringSystem
```

### 3. Создать виртуальное окружение
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 4. Установить зависимости
```bash
pip install --upgrade pip
pip install ultralytics "numpy==1.26.4" opencv-python onnxruntime "mediapipe==0.10.9"
pip install streamlit Pillow

# dlib компилируется из исходников (~10 минут)
brew install cmake dlib
pip install dlib
pip install face_recognition
```

### 5. Скачать модели
```bash
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
```

### 6. Убрать Windows-специфичный код
Откройте `app.py`, найдите и замените:
```python
# Было:
cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(self.source)

# Стало:
cap = cv2.VideoCapture(self.source)
```

---

## Запуск

### Streamlit интерфейс (Часть 2)
```bash
streamlit run app.py
```
Откройте браузер: [http://localhost:8501](http://localhost:8501)

### Консольный режим (Часть 1)
```bash
python main.py
```

---

## Добавление студентов в базу

```bash
python manage_db.py
```

Введите имя студента — система сделает 5 снимков через веб-камеру и сохранит вектор лица в `database/faces_db.pkl`.

---

## Параметры детекции

| Параметр | По умолчанию | Описание |
|---|---|---|
| Порог уверенности | 0.50 | Минимальная уверенность YOLO для фиксации объекта |
| Буфер записи | 5 сек | Пауза после исчезновения нарушения до сохранения |
| Мин. длительность | 2 сек | Минимальное время нарушения для фиксации |
| Пропуск кадров | 3 | Обрабатывать каждый N-й кадр (выше = быстрее) |
| Порог EAR | 0.22 | Порог закрытости глаз (при очках рекомендуется 0.18) |
| Время до сна | 3 сек | Сколько секунд глаза закрыты до фиксации нарушения |

---

## Выходные данные

После каждого нарушения автоматически сохраняется:

```
outputs/
├── segments/phone_usage_20260310_142301.mp4   # видеоклип нарушения
├── faces/Сириков_Вячеслав_20260310_142301.jpg # фото известного нарушителя
├── faces/Unknown_483921_20260310_142415.jpg   # фото неизвестного нарушителя
└── reports/
    ├── report_20260310_142301.json            # JSON отчёт
    └── report_20260310_142301.txt             # текстовый отчёт
```

Отчёт сохраняется автоматически при нажатии кнопки **Остановить**.

### Пример JSON отчёта

```json
{
  "date": "2026-03-10 14:23:01",
  "total": 2,
  "counts": { "phone_usage": 1, "sleeping": 1 },
  "violations": [
    {
      "type": "phone_usage",
      "conf": 0.81,
      "start": "14:03:59",
      "end": "14:04:05",
      "duration": 6,
      "person": "Сириков Вячеслав",
      "face_photo": "outputs/faces/Сириков_Вячеслав_20260310_140359.jpg",
      "segment": "outputs/segments/phone_usage_20260310_140359.mp4"
    }
  ]
}
```

---

## Известные ограничения

- На Windows при кириллице в пути к папке может не работать face_recognition — решение: скопировать модели dlib в `C:\models\`
- MediaPipe FaceMesh требует хорошего освещения для точного EAR
- На macOS при первом запуске потребуется разрешить доступ к камере

---

## Авторы

Проектная работа №6 — Система мониторинга дисциплины на занятиях
