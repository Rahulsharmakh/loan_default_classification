from flask import Flask, request, render_template, redirect, url_for  
import joblib
import numpy as np
from model_utils import load_model

model = load_model()

app=Flask(__name__)

model = joblib.load('rf_final_model.pkl')  
scaler = joblib.load('robust_scaler.pkl') 
encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['Age'])
    income = float(request.form['Income'])
    loan_amount = float(request.form['LoanAmount'])
    credit_score = int(request.form['CreditScore'])
    months_employed = int(request.form['MonthsEmployed'])
    num_credit_lines = int(request.form['NumCreditLines'])
    interest_rate = float(request.form['InterestRate'])
    loan_term = int(request.form['LoanTerm'])
    dti_ratio = float(request.form['DTIRatio'])

    education = request.form['Education']
    employment_type = request.form['EmploymentType']
    marital_status = request.form['MaritalStatus']
    has_mortgage = request.form['HasMortgage']
    has_dependents = request.form['HasDependents']
    loan_purpose = request.form['LoanPurpose']
    has_co_signer = request.form['HasCoSigner']

    # Mapping dictionaries
    education_map = {
        "Bachelor's": 0,
        "High School": 1,
        "Master's": 2,
        "PhD": 3
    }

    employment_type_map = {
        "Full-time": 0,
        "Part-time": 1,
        "Self-employed": 2,
        "Unemployed": 3
    }

    marital_status_map = {
        "Divorced": 0,
        "Married": 1,
        "Single": 2
    }

    loan_purpose_map = {
        "Auto": 0,
        "Business": 1,
        "Education": 2,
        "Home": 3,
        "Other": 4
    }

    yes_no_map = {"Yes": 1, "No": 0}

    
    education_encoded = education_map[education]
    employment_type_encoded = employment_type_map[employment_type]
    marital_status_encoded = marital_status_map[marital_status]
    loan_purpose_encoded = loan_purpose_map[loan_purpose]
    has_mortgage_encoded = yes_no_map[has_mortgage]
    has_dependents_encoded = yes_no_map[has_dependents]
    has_co_signer_encoded = yes_no_map[has_co_signer]


    numerical_features = [
        age, income, loan_amount, credit_score,
        months_employed, num_credit_lines,
        interest_rate, loan_term, dti_ratio]

    numerical_scaled = scaler.transform([numerical_features])

    final_features = np.concatenate([
    numerical_scaled[0],  
    [education_encoded, employment_type_encoded, marital_status_encoded,
     has_mortgage_encoded, has_dependents_encoded, loan_purpose_encoded,
     has_co_signer_encoded]
    ])

    final_features = np.array([final_features]) 


    prob_default = model.predict_proba(final_features)[0][1]  
    threshold = 0.4
    prediction = 1 if prob_default >= threshold else 0

    probability_percent = round(prob_default * 100, 2)

    if prediction == 1:  
        confidence_percent = round(prob_default * 100, 2)
    else:  
        confidence_percent = round((1 - prob_default) * 100, 2)


    result = "Loan is likely to DEFAULT" if prediction == 1 else "Loan is likely to be REPAID"

    return render_template('result.html', prediction_text=result,prediction_class=prediction,confidence=confidence_percent)




if __name__ == '__main__':
    app.run(debug=False)