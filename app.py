# -*- coding: utf-8 -*-
"""
Created on Tue May 17 07:43:50 2022

@author: DELL
"""

import numpy as np 
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('RandomForest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():   
     N = int(request.form['N'])
     P = int(request.form['P'])
     K = int(request.form['K'])
     Temperature = float(request.form['T'])
     Humidity = float(request.form['H'])
     ph = float(request.form['pH'])
     rainfall = float(request.form['R'])
    
     final_features = np.array([[N,P,K,Temperature,Humidity,ph,rainfall]])
     prediction = model.predict(final_features)
    
     output = prediction[0]
    
     return render_template('index.html', prediction_text = output)

if __name__ == "__main__":
    app.run(debug=True)