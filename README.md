# Travel Diary

Генератор HTML-дневника путешествия. Берёт папку с фотографиями и GPX-трек,
извлекает метаданные, группирует кадры по дням и локациям, генерирует тексты
через локальные и облачные LLM и собирает интерактивную презентацию со слайдами.

## Возможности

- **Метаданные из EXIF** — дата, GPS, модель камеры, часовой пояс (`OffsetTimeOriginal`).
- **Fallback на mtime** — если EXIF-даты нет, берётся время изменения файла.
- **Интерполяция GPS из GPX** — для фото без геометки координаты берутся из
  ближайших точек трека по времени.
- **Кластеризация по локациям** — кадры группируются по расстоянию (по умолчанию
  радиус 400 м), внутри дня выделяются major (свои слайды) и minor (transit) локации.
- **Обратное геокодирование** — Nominatim превращает координаты в названия городов.
- **Википедия по координатам** — геопоиск вместо текстового, чтобы «район Чаоян в
  Пекине» не путался с «городом Чаоян в Ляонине». Есть fallback на текстовый
  поиск, если ближайшая статья дальше `max_dist_m`.
- **Описания кадров через VLM** — локально (`minicpm-v4.5:8b`, `llava:7b`,
  `moondream`) с автоматическим fallback между моделями.
- **Тексты через облако** — Groq (`gpt-oss-120b`, `qwen3.8-27b`, `gpt-oss-20b`)
  или OpenRouter, с rate-limit и retry.
- **Вычитка (proofread)** — отдельная роль с низкой температурой, ловит иноязычные
  вставки, латиницу, транслит и опечатки.
- **Многослойный дневник** — обложка, пролог, обзор маршрута, day-intro,
  слайды локаций, transit, финал.
- **Дедупликация и диверсификация фото** — по времени, перцептивному хешу и
  географическому кластеру, чтобы на слайде не было подряд одинаковых кадров.
- **Lightbox** — клик по фото открывает крупный просмотр с навигацией по кадрам
  слайда. Под фото показывается сгенерированное описание, название локации и
  заголовок кластера.
- **Якоря и ссылки на слайд** — у каждого слайда есть `id="slide-N"`, кнопка
  «🔗 Ссылка» копирует URL прямо на текущий слайд. Открытие по хешу работает.
- **Экспорт фото** — уменьшение до `export_max_side`, вырезание всех метаданных
  и переименование в `slide_NN_MM.jpg`.

## Требования

- Python 3.10+
- Ollama (локально или на другом ПК в сети) с моделями:
  - `minicpm-v4.5:8b` — основной VLM
  - `llava:7b` — резерв VLM
  - `moondream:latest` — быстрый резерв VLM
  - `qwen2.5-coder:1.5b` — reduce (аварийное сжатие)
- Ключи Groq и/или OpenRouter (необязательно, но сильно улучшают тексты)

## Установка

```bash
git clone https://github.com/TSergeymsk/travel-diary.git
cd travel-diary

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

Если на Debian/Ubuntu нет `python3-venv`:

```bash
sudo apt install -y python3-venv python3-full
```

### Модели в Ollama

```bash
ollama pull minicpm-v4.5:8b
ollama pull llava:7b
ollama pull moondream:latest
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5:7b         # опционально, для reduce
```

### Переменные окружения

```bash
export GROQ_API_KEY="gsk_..."              # https://console.groq.com/keys
export OPENROUTER_API_KEY="sk-or-v1-..."   # https://openrouter.ai/keys
```

Можно положить в `~/.bashrc`, чтобы жило между сессиями.

## Быстрый старт

```bash
python travel_diary.py /путь/к/фото \
    --gpx /путь/к/треку.gpx \
    -o /путь/куда/собрать \
    -c config.yaml
```

Открыть результат:

```bash
xdg-open /путь/куда/собрать/index.html
```

## Структура проекта

```
travel-diary/
├── README.md
├── requirements.txt
├── config.yaml
├── providers.py             # бэкенды LLM, retry, rate limit, фазы VRAM
├── search.py                # Википедия через геопоиск
├── travel_diary.py          # основной пайплайн
├── upgrade_yaml.py          # апгрейд старого config.yaml
├── timeline_to_gpx.py       # конвертер Google Timeline → GPX
└── templates/
    └── slides.html.j2       # HTML-шаблон со слайдами и lightbox
```

## CLI

| Флаг | Описание |
|---|---|
| `photos` | папка с фотографиями (обязательный) |
| `-o, --out` | папка результата (по умолчанию `./diary`) |
| `-c, --config` | путь к `config.yaml` |
| `-g, --gpx` | файл GPX с треком |
| `--skip-vision` | пропустить анализ изображений (быстрее и дешевле) |
| `--skip-proofread` | не прогонять тексты через корректора |
| `--single-slide-days` | старый режим: один слайд на день, без разбивки по локациям |
| `--max-days N` | обработать только первые N дней (отладка) |
| `--no-phase` | отключить фазовый режим Ollama |
| `-v, --verbose` | подробный лог (DEBUG) |

## Пайплайн

1. **Метаданные** — обход всех файлов, EXIF-дата, GPS, `OffsetTimeOriginal`,
   fallback на mtime.
2. **GPX** — расчёт расстояния по дням, интерполяция GPS для фото без геометок.
3. **Кластеризация** — группировка фото по дням, внутри дня — по расстоянию.
4. **Геокодинг + Википедия** — Nominatim + Википедия по координатам.
5. **Фаза A: vision** — описания кадров. Все вызовы VLM за один прогон,
   модель загружается в VRAM один раз.
6. **Классификация** — major (свои слайды) vs minor (transit) локации.
7. **Фаза B: text + proofread** — заголовки, нарративы локаций, day-intro,
   обзор маршрута, пролог, финал. Вычитка каждого текста.
8. **Экспорт фото** — переименование в `slide_NN_MM.jpg`, вырезание EXIF,
   уменьшение до `export_max_side`.
9. **HTML** — рендер шаблона с якорями, lightbox и описаниями.

## Конфигурация

Полный пример — в `config.yaml`. Кратко по секциям.

### `runtime`

| Параметр | Описание |
|---|---|
| `ollama_host` | адрес Ollama-сервера (`http://localhost:11434` или IP другого ПК) |
| `phase_mode` | фазовый режим: сначала всё зрение, потом все тексты |
| `preload` | прогревать модель до первого запроса |
| `unload_between_phases` | выгружать модели из VRAM после фазы |
| `keep_alive` | сколько держать модель в VRAM (`30m`) |
| `retry` | параметры экспоненциального retry для облачных бэкендов |

### `roles`

Четыре роли: `vision` (локально), `text` (облако), `proofread` (облако),
`reduce` (локально). Каждая роль — цепочка бэкендов, при падении одного
переходим к следующему. Если вся цепочка упала — вызывается `fallback_role`.

Параметры одного бэкенда:

```yaml
- backend: groq                          # ollama | groq | openrouter | openai
  model: openai/gpt-oss-120b
  label: groq-120b
  api_key_env: GROQ_API_KEY
  base_url: https://api.groq.com/openai/v1
  temperature: 0.85
  max_tokens: 1200
  reasoning_effort: low                  # для gpt-oss и подобных
  rate_limit:
    rpm: 25
    tpm: 7000
    min_interval_s: 0.5
```

Для ollama-бэкендов дополнительно:

```yaml
  num_ctx: 4096                          # критично для 8 ГБ VRAM
  keep_alive: 30m
```

### `processing`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `vision_max_side` | 768 | до какого размера уменьшать фото для VLM |
| `export_max_side` | 1600 | размер фото в HTML |
| `cluster_radius_m` | 400 | радиус склейки кадров в локацию |
| `max_images_per_cluster` | 4 | сколько кадров давать VLM за раз |
| `use_reverse_geocode` | true | Nominatim |
| `use_file_mtime_fallback` | true | брать mtime, если нет EXIF-даты |
| `photo_tz_offset_hours` | 3 | часовой пояс для naive-дат |
| `max_photos_per_day` | 6 | фото на слайде дня |
| `max_photos_per_location` | 4 | фото на слайде локации |
| `dedup_time_window_s` | 5 | окно дедупликации по времени |
| `dedup_hash_threshold` | 10 | порог хемминга для перцептивного хеша |
| `max_photos_per_cluster` | 2 | не больше N фото из одного кластера |
| `significant_min_photos` | 3 | минимум фото, чтобы кластер стал major |
| `significant_require_wiki` | false | только кластеры со статьёй — major |
| `day_intro_min_locations` | 2 | day-intro при ≥ N major-локациях |
| `generate_route_overview` | true | делать слайд с обзором маршрута |
| `transit_min_clusters` | 2 | минимум minor-кластеров для transit-слайда |
| `transit_min_photos` | 3 | минимум фото в minor-кластерах |
| `gps_interpolation_from_gpx` | true | интерполировать GPS из GPX |
| `gps_interpolation_max_gap_min` | 30 | максимальный разрыв до точки трека |

### `search`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `enabled` | true | искать статьи в Википедии |
| `language` | ru | язык Википедии |
| `max_chars` | 600 | обрезать extract до N символов |
| `timeout` | 8 | таймаут запросов |
| `radius_m` | 5000 | радиус геопоиска |
| `max_dist_m` | 2000 | если ближайшая статья дальше — текстовый поиск |

### `output`

| Параметр | По умолчанию | Описание |
|---|---|---|
| `title` | «Моё путешествие» | заголовок на обложке и в `<title>` |
| `theme` | dark | `dark` или `light` |

## Ollama на другом ПК

По умолчанию Ollama слушает только `127.0.0.1`. Чтобы открыть её в локальной
сети:

**На ПК с Ollama (Windows):**

```powershell
New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound `
    -Protocol TCP -LocalPort 11434 -Action Allow
setx OLLAMA_HOST "0.0.0.0:11434"
setx OLLAMA_ORIGINS "*"
# перезапустить Ollama
Restart-Service ollama -ErrorAction SilentlyContinue
```

**Проверка с клиента:**

```bash
curl http://192.168.2.2:11434/api/tags
```

В `config.yaml`:

```yaml
runtime:
  ollama_host: http://192.168.2.2:11434
```

## Подготовка GPX

### Вариант 1: свой трекер

Любой GPX с `<trkpt>` и `<time>` внутри. Подойдёт Strava, Garmin, OsmAnd,
OwnTracks и т.п.

### Вариант 2: Google Timeline

Сначала экспортируйте `Timeline.json` из приложения Google Maps на телефоне:

1. Настройки → Местоположение → Службы определения местоположения → Хронология.
2. «Экспорт данных хронологии» → сохранить `Timeline.json`.

Затем сконвертируйте в GPX с сохранением временных меток:

```bash
python timeline_to_gpx.py Timeline.json -o timeline.gpx --tz 3
```

Параметр `--tz` — часовой пояс снимков (Москва = 3, Пекин = 8). Влияет только
на разбивку точек по календарным дням.

### Вариант 3: Google Maps маршрут

Скопируйте ссылку на маршрут (`https://www.google.com/maps/dir/...`) и
пропустите через `mapstogpx.com` или аналогичный сервис.

## Апгрейд старого config.yaml

Если конфиг остался от предыдущих версий:

```bash
python upgrade_yaml.py ~/config.yaml
```

Скрипт идемпотентен: добавит недостающие ключи и сохранит бэкап с timestamp.
Повторный запуск ничего не испортит.

## Результат

```
diary/
├── index.html               # самодостаточная презентация
└── photos/
    ├── slide_01_01.jpg      # обложка
    ├── slide_04_01.jpg      # первое фото 4-го слайда
    ├── slide_04_02.jpg      # второе фото 4-го слайда
    └── ...
```

Открывается в любом браузере, работает офлайн, метаданные из фото вырезаны,
размер каждой картинки не превышает `export_max_side`.

## Управление презентацией

| Действие | Горячая клавиша |
|---|---|
| Следующий слайд | `→`, `Space`, `PageDown` |
| Предыдущий слайд | `←`, `PageUp` |
| Первый / последний | `Home` / `End` |
| Открыть фото | клик по картинке |
| Закрыть lightbox | `Esc`, клик по фону, кнопка × |
| Листать фото | `←` / `→` внутри lightbox, кнопки ‹ и › |
| Ссылка на слайд | кнопка «🔗 Ссылка» снизу |

Ссылки вида `...index.html#slide-7` открывают дневник сразу на нужном слайде.

## Экономия токенов и времени

- `--skip-vision` отключает описания кадров, оставляя все остальные тексты
  на основе метаданных. Экономит 60–80% времени.
- `--skip-proofread` отключает вычитку. Экономит ~40% вызовов облачных API,
  но оставляет возможные латиницу и транслит в описаниях.
- `significant_min_photos: 5` уменьшает количество major-локаций, снижая
  число вызовов LLM на 30–50%.
- Для 8 ГБ VRAM: `num_ctx: 4096`, `vision_max_side: 512`,
  `max_images_per_cluster: 2`.

## Проблемы и решения

**`UnboundLocalError` в `cluster_photos`** — устаревшая версия файла.
Обновите до актуальной: `git pull`.

**Ollama не видна по сети** — проверьте `OLLAMA_HOST=0.0.0.0:11434`
на сервере и правило брандмауэра. Убедитесь, что клиент в той же подсети.

**Медленный vision (20+ сек/фото)** — модель свопится. Проверьте VRAM:
`nvidia-smi` на Windows, `curl http://ollama-host:11434/api/ps` для размера
модели. Помогает закрыть Chrome, Edge, VS Code; уменьшить `num_ctx`,
`vision_max_side`, `max_images_per_cluster`.

**`WARNING: [proofread] все бэкенды исчерпаны, fallback`** — reasoning-модель
израсходовала `max_tokens` на внутренние размышления. Убедитесь, что в
конфиге стоит `reasoning_effort: low` для всех gpt-oss и qwen3.8 бэкендов.

**403 от Википедии** — в `search.py` замените контакт в `User-Agent`
(`travel-diary/1.0 ...`) на реальный URL или email.

**`429 Too Many Requests` от Groq** — работает автоматически: retry с
экспоненциальным backoff. Если повторяется часто, понизьте `tpm` в
`rate_limit` до 6000.

**`model_not_found` от Groq** — модель устарела или недоступна в вашем
тарифе. Список актуальных смотрите в консоли Groq и правьте `config.yaml`.

**EXIF-даты без часового пояса** — если в фото нет `OffsetTimeOriginal`,
подставится `photo_tz_offset_hours` из конфига. Для поездки через несколько
зон точность ±5 часов может развести фото по разным дням. Решение —
per-photo tz через доработку или указание среднего значения.

## Лицензия

MIT.