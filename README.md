# Loan Approval Prediction

A responsive Flask web application that predicts loan approval using a saved scikit-learn preprocessing pipeline and Decision Tree classifier. Users enter applicant and financial details and receive a predicted decision and approval probability.

## Features

- Applicant details, income, credit history, loan term, and property area inputs.
- Total-income and loan-to-income feature engineering.
- Categorical preprocessing and classification in a saved pipeline.
- Responsive HTML/CSS form and probability indicator.

## Technology and structure

Python, Flask, pandas, scikit-learn, joblib, HTML, and CSS.

```text
app.py                  Flask routes and inference
model/loan_model.pkl     Supplied fitted pipeline
static/style.css        Responsive styling
templates/index.html    Form and prediction results
requirements.txt        Pinned runtime dependencies
.gitignore              Local files and generated-file exclusions
```

## Run locally

The application was smoke-tested with Python 3.13 and the dependencies pinned in `requirements.txt`.

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run from the project directory:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, complete the form, and select **Predict Loan Approval**. Debug mode is disabled. The development server is intended for local use; hosting requires a production WSGI server and deployment configuration.

## How it works

Form -> Flask `/predict` -> pandas DataFrame -> preprocessing -> Decision Tree -> result.

The app computes `Total_Income = ApplicantIncome + CoapplicantIncome` and `Loan_Income_Ratio = LoanAmount / (Total_Income + 1)`. It treats class `1` as approval and the second probability column as approval probability.

## Model and reproducibility

The supplied model contains a categorical preprocessor and `DecisionTreeClassifier`. Its metadata records scikit-learn **1.6.1**, pinned for compatibility. The small saved runtime model is intentionally included because the app requires it at startup.

The source ZIP contains no training dataset, training script, evaluation metrics, dataset attribution, or license. Training cannot be reproduced and model accuracy cannot be established from these files. Input currency and scaling are undocumented; confirm the original training units before interpreting results. No accuracy claim is made.

Only load trusted model files: joblib/pickle deserialization can execute code.

## Limitations

This is a portfolio demonstration, not a lending decision service. Gender, marital status, and other attributes may introduce bias; fairness and probability calibration have not been evaluated. There is no authentication, persistence, or production security hardening. Use synthetic examples rather than real personal or financial information.

## Future improvements

- Provide training code, dataset provenance, and reproducible evaluation.
- Evaluate fairness and probability calibration.
- Add server-side input validation and automated inference tests.
- Select a license after confirming rights to the code and model.
