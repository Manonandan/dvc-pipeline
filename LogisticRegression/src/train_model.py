import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Read the train dataset
train_data = pd.read_csv('LogisticRegression/dataset/train_data.csv')
X_train = train_data[['feature1', 'feature2']].values
y_train = train_data['label'].values

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model as a pickle file
with open('LogisticRegression/model/lr_model.pkl', 'wb') as file:
    pickle.dump(model, file)