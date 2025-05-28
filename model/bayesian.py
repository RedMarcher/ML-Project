from datainput import load_data_from_xlsx
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, matthews_corrcoef, roc_curve, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read in data
X, y = load_data_from_xlsx("target",["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Create Gaussian Naive Bayes model
model = GaussianNB()

# K-fold cross validation is used to estimate how well the model will generalize
k_folds = 5
cv = KFold(n_splits=k_folds, shuffle=True, random_state=420)
accuracy_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

print(f"Accuracy scores for each of the {k_folds} folds: {accuracy_scores}")
print(f"Mean cross-validation accuracy: {np.mean(accuracy_scores):.8f}")

# Train the model with the entire training data
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate accuracy
accuracy  = accuracy_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)

# Calculate specificity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

###MORGAN INSERT
##MORGAN WAS HERE
###MCC Matthews Correlation Coefficent(MCC)
y_score = model.predict_proba(X_test_scaled)[:,1]
mcc = matthews_corrcoef(y_test, y_pred)
##AUC
auc = roc_auc_score(y_test, y_score)


print(f"AUC: {auc}")
print(f"Mcc: {mcc : .8f}")


print(f"AUC: {auc*100: .5f}")
print(f"Mcc: {mcc : .8f}")

print(f"\nTest Set Accuracy: {accuracy:.8f}")
print(f"Precision:           {precision:.8f}")
print(f"Recall (Sensitivity):{recall:.8f}")
print(f"Specificity:         {specificity:.8f}")

# Plot the confusion matrix (Exact prediction numbers)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Heart Disease', 'Heart Disease'],
            yticklabels=['No Heart Disease', 'Heart Disease'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.savefig("bayesian_model_results.jpg")

# Model Performance Measures ( For Project's Exploratory Analysis ) -FG
# ------------------------------------------------------------------

# ( TPR - FPR ) ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_score)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
#plt.show()
plt.savefig("roc_curve.png")  # saves to file 

# ( Precision – Recall ) Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_score)
plt.figure()
plt.plot(recall_vals, precision_vals)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve')
#plt.show()
plt.savefig("precision_recall_curve.png") # saves to file 

# Predicted Probabilities ( True Class - False Class ) Distribution
plt.figure()
plt.hist(y_score[y_test == 0], bins=20, label='No Heart Disease')
plt.hist(y_score[y_test == 1], bins=20, label='Heart Disease')
plt.xlabel('Predicted Probability of Disease')
plt.ylabel('Count')
plt.title('Predicted-Probability Distribution')
plt.legend()
#plt.show()
plt.savefig("probability_distribution.png") # saves to file 

# Misclassification Rate
misclass_rate = np.mean(y_test != y_pred)
print(f"Zero–One Loss (Misclassification Rate): {misclass_rate:.8f}")





















