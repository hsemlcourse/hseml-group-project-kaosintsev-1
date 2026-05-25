[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — BNP Paribas Cardif Claims Management

**Задача:** бинарная классификация `target` (0/1)  
**Студент:** Осинцев Кирилл Андреевич, БИВ235  
**Датасет:** [Kaggle: BNP Paribas Cardif Claims Management](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)  
**Целевая метрика:** Log Loss (дополнительно ROC-AUC)

Датасет содержит анонимизированные данные о страховых заявках: быстрое одобрение (класс 1) или дополнительная проверка (класс 0).

## Структура репозитория

```text
.
├── data
│   ├── processed/          # сабмиты, промежуточные файлы
│   └── raw/                # train.csv, test.csv
├── models/                 # обученная модель и препроцессор
├── notebooks/              # EDA, baseline, эксперименты
├── report/                 # отчёт и графики
├── scripts/
│   ├── train_model.py      # обучение и сохранение модели
│   └── export_report_figures.py
├── src/
│   ├── api/main.py         # FastAPI
│   ├── app/streamlit_app.py
│   ├── config.py
│   ├── inference.py
│   ├── modeling.py
│   └── preprocessing.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── tests/
```

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Положите `train.csv`, `test.csv` в `data/raw/`.

## Обучение модели

```bash
python scripts/train_model.py
python scripts/export_report_figures.py
```

## Деплой (Docker)

```bash
docker compose up --build
```

| Сервис | URL |
|--------|-----|
| FastAPI (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |
| Jupyter | http://localhost:8888 |

### API без Docker

```bash
uvicorn src.api.main:app --reload --port 8000
streamlit run src/app/streamlit_app.py
```

## Тесты и линтер

```bash
pytest tests/ -v
ruff check src/ --line-length 120
```

## Отчёт

Полный отчёт: [`report/report.md`](report/report.md)  
