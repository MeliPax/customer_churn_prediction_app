# import Flask class from the flask module
import gzip

from flask import Flask, request, render_template

import numpy as np
import pandas as pd
import pickle

# Create Flask object to run
app = Flask(__name__)

infile = open("models/customer_churn_model.sav", 'rb')

try:
    new_dict = pickle.load(infile)
except:
    print("An exception occurred")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global testData
    if request.method == 'POST':
        # Get values from browser
        tenure = int(request.form['tenure'])
        m_charges = int(request.form['m_charges'])
        tot_charges = int(request.form['tot_charges'])
        s_citizen = int(request.form['s_citizen'])
        partner = int(request.form['partner'])
        dependents = int(request.form['dependents'])
        contract = int(request.form['contract'])
        tec_support = int(request.form['tec_support'])
        online_sec = int(request.form['online_sec'])
        device_prot = int(request.form['device_prot'])
        pl_billing = int(request.form['pl_billing'])
        payment_m = int(request.form['payment_m'])
        online_backup = int(request.form['online_backup'])

        input_variables = pd.DataFrame([[tenure, m_charges, tot_charges, s_citizen, partner, contract, tec_support,
                                         online_sec, online_backup, device_prot, pl_billing, payment_m, dependents]],
                                       columns=['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Partner',
                                                'Contract', 'TechSupport', 'OnlineSecurity', 'OnlineBackup',
                                                'DeviceProtection', 'PaperlessBilling',
                                                'PaymentMethod', 'Dependents'], dtype=int)
        # print(input_variables)
        # print(new_dict)
        predicted = int(new_dict.predict(input_variables))
        print("predicted")
        print(predicted)
        if predicted == 1:
            churn = "Yes"
        else:
            churn = "No"
        output = "Customer Churn: " + str(predicted)

        return output


if __name__ == "__main__":
    print("**Starting Server...")
    # Call function that loads Model
    app.debug = True
    # Run Server
    app.run(host='127.0.0.1', port=5000)
