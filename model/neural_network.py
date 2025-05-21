import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data with piping
features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
df = pd.read_csv("../data/heart_og.csv") # I Run this inside the model folder; if you run in a different directory, change this.
df = df.loc[:, df.nunique() > 1] # drop any constant columns, remove features with only a single unique value
df = df.dropna(subset=features + ["target"])
features = [f for f in features if f in df.columns]
X = df[features].values
y = df["target"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("NaNs in X_train_scaled:", np.isnan(X_train_scaled).sum())
print("Infs in X_train_scaled:", np.isinf(X_train_scaled).sum())


# Create MLPClassifier as an NN model
# one hidden layer of 100 neurons
# max_iter increased to ensure convergence
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=69
)

# K-fold cross validation is used to estimate how well the model will generalize
k_folds = 5
cv = KFold(n_splits=k_folds, shuffle=True, random_state=420)
accuracy_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

# Train the model with the entire training data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate accuracy
accuracy    = accuracy_score(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)
precision   = precision_score(y_test, y_pred)
recall      = recall_score(y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

##MORGAN WAS HERE
###MCC Matthews Correlation Coefficent(MCC)
y_score = model.predict_proba(X_test_scaled)[:,1]
mcc = matthews_corrcoef(y_test, y_pred)
##AUC
auc = roc_auc_score(y_test, y_score)


print(f"AUC: {auc}")
print(f"Mcc: {mcc : .8f}")



print(f"\nTest Set Accuracy:  {accuracy:.8f}")
print(f"Precision:            {precision:.8f}")
print(f"Recall (Sensitivity): {recall:.8f}")
print(f"Specificity:          {specificity:.8f}")

# Plot the confusion matrix (Exact prediction numbers)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Neural Network Confusion Matrix')
plt.savefig("nn_model_results.jpg")

