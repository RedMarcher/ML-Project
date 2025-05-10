import os
from flask import Flask, render_template, request
from services import *
#import your_ml_model  # Replace with your machine learning model import

# Load your pre-trained machine learning model here
#model = your_ml_model.load_model()  # Replace with your model loading logic

HERE = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.abspath(os.path.join(HERE, "..", "templates"))

app = Flask(__name__, template_folder=TEMPLATE_FOLDER)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = []
        #request.form.get("input")


        #(Add each 13 features)
        input_data.append(request.form.get('age'))
        input_data.append(request.form.get('sex'))

        ##Deal w/ checkbox for model type (14th input)
        modeltype = 'ANN'

        result = {}
        input_data = [42,1,2,130,180,0,1,150,0,0,2,0,2]
        if modeltype == 'ANN':
            result['ANN'] = predict_ANN(input_data)

        return render_template("result.html", prediction=result)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)