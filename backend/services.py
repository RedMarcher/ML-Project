import joblib
from sklearn.preprocessing import StandardScaler

def predict_ANN(X_input):
    trainedANN, trainedScaler = joblib.load("../model/trainedModels/trainedANN.joblib")
    test_point = [X_input]
    test_point_scaled = trainedScaler.transform(test_point)
    return trainedANN.predict(test_point_scaled)
