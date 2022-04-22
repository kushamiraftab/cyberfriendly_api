# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to the cyber-bullying detection API!</h1>"


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name")

    # For debugging
    print(f"Name received")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No name found. Please send a name."
    # Check if the user entered a number
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)



@app.route('/predict/', methods=['POST'])
def predict():

    # Load the required models and data
    vectorizer = joblib.load("./cyberb_vectorizer.pkl")
    print('Vectorizer loaded')
    model = joblib.load("./cyberb_model.pkl")
    print('Model loaded')
    cols = joblib.load("./cyberb_columns.pkl")
    print('Model columns loaded')

    if model and vectorizer and cols:
        try:
            # Extract the queries from json file
            json_ = request.json
            txtlst = json_[0]['text']
            text_pro = []
            for text in txtlst:
                text_token = vectorizer.transform([text]).toarray()
                num_words = len(text.split(' '))
                full = np.append(text_token, num_words)
                text_pro.append(full)

            final_df = pd.DataFrame(text_pro, columns = cols)

            # run the predictions on that dataframe
            prediction = model.predict(final_df)
            maphash = {2: 'No cyberbullying', 1: 'Cyberbullying detected', 0: 'Cyberbullying detected'}

            result = []
            for pred in prediction:
                result.append(maphash[pred])

            return jsonify({'prediction': result})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return('No model here to use')


if __name__ == '__main__':
    app.run(port=5000, threaded = True, debug=True)