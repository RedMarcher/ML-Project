import joblib
from sklearn.preprocessing import StandardScaler

def predict_ANN(X_input):
    trainedANN = joblib.load("./trainedModels/trainedANN.joblib")
    test_point = [X_input]
    scaler = StandardScaler()
    test_point_scaled = scaler.transform(test_point)
    return trainedANN.predict(test_point_scaled)
