import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import warnings

warnings.filterwarnings("ignore")

# LOADING DATA FOR BBP
data = pd.read_csv("Pathoprophet/bbp_top7_symptom_pathogen.csv")
data = np.array(data)

X = data[:, :-1]
y = data[:, -1]

X = X.astype('int')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM model
svm = SVC(probability=True, kernel='rbf', C=1, gamma='scale')
svm.fit(X_train_scaled, y_train)

# Evaluate accuracy
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"BBP Model Accuracy: {accuracy:.4f}")

# Example input
inputt = [int(x) for x in "1 0 1 0 1 1 1 0 1 1 1 1 0 0 0 1 1 0 1 0 1".split()]
final = scaler.transform([np.array(inputt)])

# Predict probabilities
b = svm.predict_proba(final)

# Save the model
pickle.dump(svm, open('svm_model_BBP.pkl', 'wb'))

# Load the model
model = pickle.load(open('svm_model_BBP.pkl', 'rb'))



# # LOADING DATA FOR URTI
# data = pd.read_csv("urtisymptoms.csv")
# data = np.array(data)

# X = data[:, :-1]
# y = data[:, -1]

# X = X.astype('int')

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Create and train SVM model
# svm = SVC(probability=True, kernel='rbf', C=1, gamma='scale')
# svm.fit(X_train_scaled, y_train)

# # Evaluate accuracy
# y_pred = svm.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# print(f" URTI Model Accuracy: {accuracy:.4f}")

# # Example input
# inputt = [int(x) for x in "1 0 1 0 1 1 1 0 1 1 1 1 0 0 0 1 1 0 1 0 1".split()]
# final = scaler.transform([np.array(inputt)])

# # Predict probabilities
# b = svm.predict_proba(final)

# # Save the model
# pickle.dump(svm, open('svm_model_URTI.pkl', 'wb'))

# # Load the model
# model = pickle.load(open('svm_model_URTI.pkl', 'rb'))















