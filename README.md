[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — BNP Paribas Cardif Claims Management

**Задача:** бинарная классификация `target` (0/1)  
**Студент:** Осинцев Кирилл Андреевич БИВ235
**Датасет:** [Kaggle: BNP Paribas Cardif Claims Management](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)  
**Целевая метрика:** Log Loss (дополнительно ROC-AUC)

Датасет содержит анонимизированные данные о страховых заявках, которые делятся на два класса: заявки, которые можно одобрить быстро (и сразу выплатить), и заявки, требующие дополнительной проверки перед одобрением

## Структура репозитория
```text
.
├── data
│   ├── processed
│   │   └── .gitkeep
│   └── raw
│       └── .gitkeep
├── models
│   └── .gitkeep
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiments.ipynb
├── presentation
│   └── README.md
├── report
│   ├── images
│   │   └── .gitkeep
│   └── report.md
├── src
│   ├── config.py
│   ├── modeling.py
│   └── preprocessing.py
├── tests
│   ├── test.py
│   └── test_preprocess.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Быстрый старт
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные
Сырые файлы (`train.csv`, `test.csv`, `sample_submission.csv`) положите в `data/raw/`.

## Обработка данныхн
- `src/preprocessing.py` - загрузка, дедупликация по `id`, импутация, feature engineering, `LabelEncoder` для категорий.
- Константы и корень проекта: `src/config.py` (`RANDOM_STATE = 42`).

## Моделирование и эксперименты
- `src/modeling.py` — стратифицированный split 70/15/15, CV Log Loss, `RandomizedSearchCV` для LightGBM, сохранение сабмита.
- `notebooks/03_experiments.ipynb` — те же шаги с таблицей кандидатов и графиком важности признаков в `report/images/`.

## Ноутбуки
- `notebooks/01_eda.ipynb` — EDA и ранние эксперименты.
- `notebooks/02_baseline.ipynb` — Logistic Regression на подготовленных данных.
- `notebooks/03_experiments.ipynb` — CV, подбор гиперпараметров LightGBM, сабмит, feature importance.

## Тесты
```bash
pytest tests/ -v
```

## Отчёт
Файл отчёта: [`report/report.md`](report/report.md)
