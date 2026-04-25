# ML Project — BNP Paribas Cardif Claims Management

**Задача:** бинарная классификация `target` (0/1)  
**Датасет:** [Kaggle: BNP Paribas Cardif Claims Management](https://www.kaggle.com/c/bnp-paribas-cardif-claims-management)  
**Целевая метрика:** Log Loss (дополнительно ROC-AUC)

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
│   ├── modeling.py
│   └── preprocessing.py
├── tests
│   └── test.py
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

`notebooks/01_eda.ipynb` также умеет читать CSV из корня проекта (если они лежат рядом с `README.md`).

## Ноутбуки
- `notebooks/01_eda.ipynb` — EDA, подготовка данных, базовые модели, PCA, сохранение сплитов
- `notebooks/02_baseline.ipynb` — заготовка под baseline этап
- `notebooks/03_experiments.ipynb` — заготовка под эксперименты

## Отчёт
Файл отчёта: [`report/report.md`](report/report.md)
