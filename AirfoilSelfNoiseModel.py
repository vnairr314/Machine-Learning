import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('airfoil_self_noise.dat', sep = '\t', header = None)

# Print statements to obtain visibility and understanding of the data
print(df.info)
print(df.head())
print(df.shape)

# The input data is contained in the first 5 columns of the data set
data = df[[0, 1, 2, 3, 4]].values

# The output data is contained in the 6th column of the data set
target = df[[5]].values

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.33)

# Creating a model using the LinearRegression class
model = LinearRegression()
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

predictions = model.predict(X_test)
#print(predictions)

# Creating a model using the RandomForestRegressor class

model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))

predictions2 = model2.predict(X_test)
print(predictions2)






