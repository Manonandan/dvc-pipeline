import pandas as pd
from sklearn.model_selection import train_test_split

# Read data from CSV file
data = pd.read_csv('dataset/synthetic_data.csv')
X = data[['feature1', 'feature2']].values
y = data['label'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save train dataset
train_data = pd.DataFrame()
train_data["feature1"] = [x[0] for x in X_train.tolist()]
train_data["feature2"] = [x[1] for x in X_train.tolist()]
train_data["label"] = y_train.tolist()

train_data.to_csv('RandomForestClassifier/dataset/train_data.csv', index=False)

# save test dataset
test_data = pd.DataFrame()
test_data["feature1"] = [x[0] for x in X_test.tolist()]
test_data["feature2"] = [x[1] for x in X_test.tolist()]
test_data["label"] = y_test.tolist()

test_data.to_csv('RandomForestClassifier/dataset/test_data.csv', index=False)
