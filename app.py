from flask import Flask, request, jsonify, render_template
#import tensorflow as tf
import numpy as np
import json
import joblib


app = Flask(__name__)

@app.before_first_request
def load_model_to_app():
    """
    Executed during the initialization of the application.
    Load neural network form the file
    """
    app.predictor = joblib.load('./static/model/lr_model.sav')


@app.route("/")
def index():
    return render_template('index.html', pred = 0)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for prediction"""

    # load encoding
    with open('./static/model/encoding.json') as json_file:
        encoding = json.load(json_file)

    data = [encoding['location_type'][request.form['location_type']],
            encoding['cellphone_access'][request.form['cellphone_access']],
            encoding['gender_of_respondent'][request.form['gender_of_respondent']], 
            encoding['education_level'][request.form['education_level']],
            encoding['job_type'][request.form['job_type']]
            ]
    data = np.array([np.asarray(data, dtype=float)])

    predictions = app.predictor.predict(data)
    print('INFO Predictions: {}'.format(predictions))

    class_dict = {0: "No", 1:"Yes"}
    class_ = class_dict[predictions[0]]

    return render_template('index.html', pred=class_)


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8000, debug=False)


if __name__ == '__main__':
    main()