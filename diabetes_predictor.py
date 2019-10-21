from flask import Flask, jsonify, request
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as Scaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    message = {'id':123, 'name':'Flask test'}
    #return 'Hello World'
    return jsonify(message)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # load the model from disk
    filename = './Models/diabetes.model'
    svc = joblib.load(filename)
    
    # load the scaler from disk
    filename = './Models/scaler.scaler'
    scaler = joblib.load(filename)

    content = request.get_json(force=True)
    print(content)

    NumTimesPrg = content['NumTimesPrg']
    print(NumTimesPrg)
    PlGlcConc = content['PlGlcConc']
    BloodP = content['BloodP']
    SkinThick = content['SkinThick']
    TwoHourSerIns = content['TwoHourSerIns']
    BMI = content['BMI']
    DiPedFunc = content['DiPedFunc']
    Age = content['Age']
 
    #Make a Prediction
    # We create a new (fake) person having the three most correated values high
    new_df = pd.DataFrame([[NumTimesPrg, PlGlcConc, BloodP,SkinThick, TwoHourSerIns, BMI, DiPedFunc, Age]])
    # We scale those values like the others
    new_df_scaled = scaler.transform(new_df)
    
    # We predict the outcome
    prediction = svc.predict(new_df_scaled)
    
    # A value of "1" means that this person is likley to have type 2 diabetes
    diagnostic = prediction[0]
    
    return jsonify({'diabetes':str(diagnostic)})
    
if __name__ == "__main__":
    app.run()
