import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

# Read the test dataset
test_data = pd.read_csv('RandomForestClassifier/dataset/test_data.csv')
X_test = test_data[['feature1', 'feature2']].values
y_test = test_data[['label']].values

# Load the model from the pickle file
with open('RandomForestClassifier/model/rfc_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Test the loaded model on test data
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the loaded model: {accuracy:.2f}")
