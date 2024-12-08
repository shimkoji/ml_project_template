├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── adhoc              <- Ad-hoc scripts
│
├── config             <- Configuration files
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── datamart        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data_processor           <- Scripts to process data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── base  <- Definition of abstract classes
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   ├── lightgbm  <- Model logic (preprocessing, prediction, etc.)
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │
│   └── optimizer      <- Scripts to optimize models
│   │   ├── base
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── preprocess.py
│   │   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│   │   └── visualize.py
│   └── utils          <- Utility scripts
│       └── __init__.py
│
└── uv.lock            <- Lock file for uv
│
└── pyproject.toml     <- Configuration file for uv
│
└── .python-version    <- Python version