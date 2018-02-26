# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('1-prostate-training-data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

y_train = y_train.reshape((160,1))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

j = 23
# Predicting the Test set results
y_pred = regressor.predict(X_test)
print("SKLearn")
print(y_pred[j][0])
print(y_test[j])


# Loss function
#W = np.random.rand(8, 1)
W = np.zeros((8, 1))

learning_rate = 0.1


for i in range(100000):
	dW = np.matmul(X_train.T, (np.matmul(X_train, W) - y_train)) * (1.0 / 160)
	#print((np.matmul(X_train, W).reshape(160,1)).shape)
	#print((np.matmul(X_train, W).reshape(160,1) - y_train).shape)
	#print(W.shape)
	#print(dW.shape)
	W -= learning_rate * dW
	loss = np.sum((np.matmul(X_train, W) - y_train) ** 2)
	#print("Loss:", loss)

# Testing
print("Hand")
y_pred = np.matmul(X_test[j], W)
print("Predicted:", y_pred[0])
print("Actual:", y_test[j])