import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#create data # vertical format The reshape(-1, 1) ensures the data is formatted correctly as a column vector, which is required by the LinearRegression model.
time_studied = np.array([20, 50, 32, 65, 23, 43,10, 5, 22, 35, 29, 5, 56]).reshape(-1,1)
scores = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57,4,89]).reshape(-1,1)
# print(time_studied)
# Uncomment this FIRST
# #train the model
# model = LinearRegression()
# model.fit(time_studied, scores)
# print(model.predict(np.array([56]).reshape(-1,1)))
# #visualize
# plt.scatter(time_studied, scores)
# plt.plot(np.linspace(0,70, 100), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
#
# plt.ylim(0,100)
# plt.show()

#Uncomment below SECOND
time_train, time_test, score_train, score_test = train_test_split(time_studied, scores, test_size=0.3)
print("time_train: ",time_train)
print("time_test: ",time_test)
print("score_train: ",score_train)
print("score_test: ",score_test)
model = LinearRegression()
model.fit(time_train, score_train)
# Evaluate the model
print("Model R^2 score:", model.score(time_test, score_test))

# Plot the regression line
plt.scatter(time_studied, scores, color='blue', label='Data points')  # Scatter plot of data points
plt.plot(np.linspace(0, 70, 100), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r', label='Regression line')  # Red regression line
plt.xlabel("Time Studied")
plt.ylabel("Scores")
plt.title("Linear Regression: Time Studied vs. Scores")
plt.legend()
plt.show()