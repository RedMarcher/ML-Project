from flask import Flask, render_template, request
#import your_ml_model  # Replace with your machine learning model import

app = Flask(__name__)

# Load your pre-trained machine learning model here
#model = your_ml_model.load_model()  # Replace with your model loading logic

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
        input_data.append(int(request.form.get('age')))
        input_data.append(int(request.form.get('sex')))

        ##Deal w/ checkbox for model type (14th input)
        modeltype = 'ANN'

        result = {}

        if modeltype == 'ANN':
            result['ANN'] = predict_ANN(input_data)

        

        number_data = request.form.get("number")

        # Preprocess the input data for your model (if needed)
        # ... your data preprocessing code here ...

        # Make prediction using your model
        #prediction = model.predict([preprocessed_data])  # Assuming a list input

        # Format the prediction for display
        #predicted_class = prediction[0]  # Assuming single class output

        #Temp
        predicted_class = "cats "
        total_animals = predicted_class * int(number_data)

        return render_template("result.html", prediction=total_animals)

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)