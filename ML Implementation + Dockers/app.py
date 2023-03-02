import pickle
#import flask
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
# Model and Scale load
regmodel = pickle.load(open('reg_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    lst = np.array(list(data.values())).reshape(1,-1)
    print(lst)
    new_data = scaler.transform(lst)
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).rechape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text = "La prediction du prix est {}".format(output))

if __name__=='__main__':
    app.run(debug = True)

