# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

def first_preprocessor(s):
    # convert to lowercase (which CountVectorizer and TfidfVectorizer do by default)
    s = s.lower()
    # replace "&amp" with "and"
    s = s.replace("&amp", "and")
    # replace multiple consecutive blank spaces with 1 blank space?
    s = re.sub("[ ]+", " ", s)
    # remove all numbers?
    s = re.sub(r'\d+', '', s)
    return(s)


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Welcome to the cyber-bullying detection API!</h1>"




@app.route('/predict/', methods=['POST'])
def predict():
    if model and vectorizer and cols:
        try:
            # Extract the queries from json file
            json_ = request.json
            print(json_)

            text_pro = []
            for q in json_:
                text = q['text']
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
    vectorizer = joblib.load("./cyberb_vectorizer.pkl")
    print('Vectorizer loaded')
    model = joblib.load("./cyberb_model.pkl")
    print('Model loaded')
    cols = joblib.load("./cyberb_columns.pkl")
    print('Model columns loaded')

    app.run(port=5000, debug=True)