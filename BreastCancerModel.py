import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = load_breast_cancer()

# Print statements to obtain more visibility and understanding of the data

print(type(data))
print(data.keys())
print(data.data)
print(data.data.shape)
print(data.target_names)
print(data.feature_names)

# Splitting data into training set and testing set
# We will train our model using the training set and then validate the training using the testing set

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
print(predictions)

N = len(y_test)

print(f'Accuracy of the predictions = {np.sum((predictions == y_test) / N)*100:.3f}%')