# Mediload — Emergency Arrivals Forecasting

A FastAPI + scikit-learn web app that predicts next-hour emergency department
arrivals from recent arrival history and calendar context.

## Project structure

```
mediload-project/
├── app/
│   ├── main.py                 # FastAPI app: routes + pipeline loading
│   ├── schemas.py               # Pydantic request/response models
│   ├── model/
│   │   └── mediload_best_pipeline.pkl   # trained sklearn Pipeline
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/style.css
│       └── js/script.js
├── train_and_save_model.py     # reproduces the notebook's training steps
├── Mediload_Emergancy.csv
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit http://127.0.0.1:8000
Swagger docs: http://127.0.0.1:8000/docs
