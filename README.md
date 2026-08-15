# MENTAL AI — Intelligent Recommendation System for E-Learning

MENTAL AI is a Streamlit-based personalized learning platform.

## Core features

- Demo assessment using `data/questions.csv`
- Personalized assessment from uploaded PDF study material
- PDF text extraction
- Study-material concept detection and analysis
- Personalized MCQ generation
- Performance and topic analysis
- Personalized learning recommendations
- Learning roadmap
- Search/questions over uploaded study material

## Technology stack

- Python
- Streamlit
- Pandas
- PyPDF
- Scikit-learn
- TF-IDF
- Cosine similarity

## Project structure

```text
MENTAL AI/
├── app.py
├── recommendation.py
├── requirements.txt
├── .gitignore
└── data/
    ├── courses.csv
    └── questions.csv
```

> `recommendation.py` is part of the working project and must be kept beside `app.py`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run MENTAL AI:

```powershell
streamlit run app.py
```

## Assessment modes

### Demo Assessment
Uses the built-in question dataset and is intended for demonstrations.

### Personalized Assessment
The student uploads study material. MENTAL AI extracts the text, analyzes concepts, generates an assessment, evaluates performance, and creates a learning roadmap.

## Important

Do not commit the virtual environment, secrets, cache files, or temporary generated Python files to GitHub.
