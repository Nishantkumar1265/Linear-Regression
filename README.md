# Importing libraries

import numpy as np
import matplotlib.pyplot as plt


# Generating data

y = 2x-5+e, where e=epsilon

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)
plt.scatter(x, y);
plt.show()

# Applying Least Square Estimation

X_mean = np.mean(x)
Y_mean = np.mean(y)

num = 0
den = 0
for i in range(len(x)):
    num += (x[i] - X_mean)*(y[i] - Y_mean)
    den += (x[i] - X_mean)**2
m = num / den #Beta1
c = Y_mean - m*X_mean #Beta0

print (m, c)

Y_pred = m*x + c

plt.scatter(x, y) # actual
plt.scatter(x, Y_pred, color='red')
plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()



# Performance Metrics

from sklearn.metrics import r2_score
r2_score(y, Y_pred) #r2_score(y_true, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y, Y_pred)

# Using Sklearn library

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(x[:, np.newaxis], y)

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

yfit=model.predict(x[:, np.newaxis])

r2_score(y,yfit)



mean_squared_error(y,yfit)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])


plt.scatter(xfit, yfit)
plt.plot(xfit, yfit);



type(x)

x.shape

x.ndim

len(x)

x1=x[:, np.newaxis]

type(x1)

x1.ndim

x1.shape

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)



# Performance Metrics



from sklearn.metrics import mean_squared_error
mean_squared_error(y, yfit)

from sklearn.metrics import r2_score
r2_score(y, yfit)

# using datasets

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

type(diabetes_X)

diabetes_X.shape
