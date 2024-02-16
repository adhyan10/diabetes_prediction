from flask import Flask,render_template,jsonify,request
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

application = Flask("__name__")
app =application

scaler = pickle.load(open("Models/standard_scaler.pkl","rb"))
log_reg = pickle.load(open("Models/logistic_reg.pkl","rb"))

##Route for homepage

@app.route("/")
def index():
    return render_template("index.html")

##Route to make predictions

@app.route("/predictdata",methods = ['GET','POST'])
def predict_datapoint():
    result = ""
    if request.method == 'POST':
        pregnancies = int(request.form.get('Pregnancies'))
        glucose = float(request.form.get('Glucose'))
        blood_pressure = float(request.form.get('BloodPressure'))
        skin_thickness = float(request.form.get('SkinThickness'))
        insulin = float(request.form.get('Insulin'))
        bmi = float(request.form.get('BMI'))
        diabetes_pedigree_function = float(request.form.get('DiabetesPedigreeFunction'))
        age = float(request.form.get('Age'))

        scaled_data = scaler.transform([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age]])
        prediction = log_reg.predict(scaled_data)
        pred_proba = log_reg.predict_proba(scaled_data)

        if prediction[0] == 1:
            result = "Diabetic"
        else:
            result = "Non-Diabetic"
        return render_template("single_prediction.html",result = result,result_proba = pred_proba[0])


    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0")