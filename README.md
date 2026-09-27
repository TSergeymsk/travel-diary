markdown
# Travel Diary

Генерирует HTML-дневник путешествия со слайдами: анализ EXIF + GPX,
описания фотографий через локальные и облачные VLM, рассказы о днях
через LLM, обогащение статьями из Википедии.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Модели в Ollama
bash
ollama pull minicpm-v4.5:8b        # vision
ollama pull llava:7b               # vision fallback
ollama pull qwen2.5:7b             # text
ollama pull qwen2.5-coder:1.5b     # reduce
Переменные окружения
bash
export GROQ_API_KEY="gsk_..."              # https://console.groq.com/keys
export OPENROUTER_API_KEY="sk-or-..."      # https://openrouter.ai/keys
Если ключей нет — удалите соответствующие ветки из config.yaml.

Запуск
bash
python travel_diary.py ~/Pictures/Trip2025 \
    --gpx ~/tracks/trip.gpx \
    -o ~/diary \
    -c config.yaml

xdg-open ~/diary/index.html
Ключи командной строки
Флаг	Значение
photos	папка с фотографиями (обязательный)
-g, --gpx	файл GPX с треком
-o, --out	куда писать результат (по умолчанию ./diary)
-c, --config	путь к конфигу (по умолчанию config.yaml)
--skip-vision	пропустить анализ изображений
--max-days N	обработать только первые N дней (отладка)
--no-phase	отключить фазовый режим (не рекомендую)
text

---

## `requirements.txt`
pillow>=10.0
exifread>=3.0
gpxpy>=1.6
pyyaml>=6.0
jinja2>=3.1
httpx>=0.27
ollama>=0.4
openai>=1.50

text

---

## `config.yaml`

```yaml
runtime:
  phase_mode: true               # сначала вся vision-фаза, потом вся text-фаза
  preload: true                  # прогревать модель пустым запросом
  unload_between_phases: true    # выгружать модели после фазы
  keep_alive: 30m                # сколько держать модель в VRAM внутри фазы

# ---- Поведение при падении цепочек ----
fallback:
  enabled: true
  # имя роли, которая делает сжатие (её собственная цепочка, без рекурсии)
  reduce_role: reduce
  # сколько первых предложений оставить, если даже reduce не справился
  hard_truncate_sentences: 3
  # авто-сжатие длинных ответов (0 = выключено)
  auto_reduce_over_chars: 0

roles:

  # ============ Vision: описания кадров ============
  vision:
    fallback_role: reduce                       # при провале всей цепочки
    empty_result: "Кадры этого места"           # что вернуть, если не вышло
    chain:
      - backend: ollama
        model: minicpm-v4.5:8b
        label: local-minicpm
        keep_alive: 30m
        temperature: 0.4
        max_tokens: 400

      - backend: ollama
        model: llava:7b
        label: local-llava
        keep_alive: 30m
        temperature: 0.4
        max_tokens: 400

      - backend: ollama
        model: moondream:latest
        label: local-moondream
        temperature: 0.3
        max_tokens: 300

      - backend: groq
        model: llama-3.2-11b-vision-preview
        label: groq-vision
        api_key_env: GROQ_API_KEY
        base_url: https://api.groq.com/openai/v1
        max_tokens: 400

      - backend: openrouter
        model: qwen/qwen-2.5-vl-72b-instruct:free
        label: openrouter-qwen-vl
        api_key_env: OPENROUTER_API_KEY
        base_url: https://openrouter.ai/api/v1
        max_tokens: 400

  # ============ Text: рассказ ============
  text:
    fallback_role: reduce
    empty_result: ""
    chain:
      - backend: ollama
        model: qwen2.5:7b
        label: local-qwen2.5
        keep_alive: 30m
        temperature: 0.85
        max_tokens: 1200

      - backend: groq
        model: llama-3.3-70b-versatile
        label: groq-70b
        api_key_env: GROQ_API_KEY
        base_url: https://api.groq.com/openai/v1
        temperature: 0.85
        max_tokens: 1200

      - backend: openrouter
        model: meta-llama/llama-3.3-70b-instruct:free
        label: openrouter-70b
        api_key_env: OPENROUTER_API_KEY
        base_url: https://openrouter.ai/api/v1
        temperature: 0.85
        max_tokens: 1200

      - backend: ollama
        model: qwen3.5:9b
        label: local-qwen3.5
        temperature: 0.85
        max_tokens: 1200

  # ============ Reduce: аварийное сжатие ============
  # У этой роли нет fallback_role — иначе возможна рекурсия.
  reduce:
    empty_result: ""
    chain:
      - backend: ollama
        model: qwen2.5-coder:1.5b
        label: local-tiny
        keep_alive: 5m
        max_tokens: 250

      - backend: ollama
        model: qwen2.5:7b
        label: local-qwen2.5
        max_tokens: 250

# ============ Параметры обработки ============
processing:
  vision_max_side: 768            # до какого размера уменьшать фото для VLM
  export_max_side: 1600           # размер фото в HTML
  cluster_radius_m: 400           # радиус склейки кадров в "место"
  max_images_per_cluster: 4       # сколько кадров на кластер отдавать VLM
  use_reverse_geocode: true       # Nominatim (нужен интернет)

# ============ Обогащение из интернета ============
search:
  enabled: true
  language: ru
  max_chars: 600
  timeout: 8

# ============ Вывод ============
output:
  title: "Моё путешествие"
  theme: dark                     # dark | light
