<pre>
.
├── README.md
├── adhoc
│   └── eda.py
├── configs
│   └── default.json
├── data
│   ├── datamart
│   ├── external
│   ├── processed
│   └── raw
│       ├── gender_submission.csv
│       ├── test.csv
│       └── train.csv
├── models
├── notebooks
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   ├── test
│   │   │   └── events.out.tfevents
│   │   ├── test_error.tsv
│   │   ├── time_left.tsv
│   │   └── tmp
│   └── eda.ipynb
├── pyproject.toml
├── references
├── reports
│   └── figures
├── run.py
├── src
│   ├── __init__.py
│   ├── data_processor
│   │   ├── __init__.py
│   │   ├── process.py
│   │   └── schema.py
│   ├── models
│   │   ├── base
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── catboost
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── lightgbm
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── predict_model.py
│   │   ├── train_model.py
│   │   └── xgboost
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── preprocess.py
│   ├── optimizer
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── catboost.py
│   │   ├── lightgbm.py
│   │   └── xgboost.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── logger.py
│   └── visualization
│       └── visualize.py
└── uv.lock
<pre>