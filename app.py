from pathlib import Path

from flask import Flask, render_template, request

import pandas as pd
import joblib


app = Flask(__name__)


# ---------------------------------------
# Load model
# ---------------------------------------

model = joblib.load(
    Path(__file__).resolve().parent / "model" / "loan_model.pkl"
)


# ---------------------------------------
# Home page
# ---------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None,
        probability=None
    )


# ---------------------------------------
# Prediction
# ---------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -------------------------------
        # Get form values
        # -------------------------------

        gender = request.form["gender"]

        married = request.form["married"]

        dependents = request.form["dependents"]

        education = request.form["education"]

        self_employed = request.form[
            "self_employed"
        ]

        applicant_income = float(
            request.form[
                "applicant_income"
            ]
        )

        coapplicant_income = float(
            request.form[
                "coapplicant_income"
            ]
        )

        loan_amount = float(
            request.form[
                "loan_amount"
            ]
        )

        loan_term = float(
            request.form[
                "loan_term"
            ]
        )

        credit_history = float(
            request.form[
                "credit_history"
            ]
        )

        property_area = request.form[
            "property_area"
        ]


        # --------------------------------
        # Feature engineering
        # --------------------------------

        total_income = (
            applicant_income +
            coapplicant_income
        )

        loan_income_ratio = (
            loan_amount /
            (total_income + 1)
        )


        # --------------------------------
        # Create DataFrame
        # --------------------------------

        input_data = pd.DataFrame({

            "Gender": [gender],

            "Married": [married],

            "Dependents": [dependents],

            "Education": [education],

            "Self_Employed": [
                self_employed
            ],

            "ApplicantIncome": [
                applicant_income
            ],

            "CoapplicantIncome": [
                coapplicant_income
            ],

            "LoanAmount": [
                loan_amount
            ],

            "Loan_Amount_Term": [
                loan_term
            ],

            "Credit_History": [
                credit_history
            ],

            "Property_Area": [
                property_area
            ],

            "Total_Income": [
                total_income
            ],

            "Loan_Income_Ratio": [
                loan_income_ratio
            ]
        })


        # --------------------------------
        # Prediction
        # --------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = (
            model.predict_proba(
                input_data
            )[0][1] * 100
        )


        # --------------------------------
        # Result
        # --------------------------------

        if prediction == 1:

            result = "Loan Approved"

        else:

            result = "Loan Rejected"


        return render_template(
            "index.html",
            prediction=result,
            probability=round(
                probability,
                2
            )
        )


    except Exception:

        app.logger.exception("Prediction failed")

        return render_template(
            "index.html",
            prediction="Unable to process this application. Please check your inputs.",
            probability=None
        )


# ---------------------------------------
# Run application
# ---------------------------------------

if __name__ == "__main__":

    app.run(
        debug=False
    )