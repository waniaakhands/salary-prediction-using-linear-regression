# TASK 1
# Experience vs Salary
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


exp = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
salary = np.array([20000,25000,30000,35000,40000,45000,50000,55000,60000,65000])

model1 = LinearRegression()
model1.fit(exp, salary)

pred1 = model1.predict([[5]])

print("Task 1 Prediction (5 years):", pred1[0])

mse1 = mean_squared_error(salary, model1.predict(exp))
r21 = r2_score(salary, model1.predict(exp))

print("MSE:", mse1)
print("R2 Score:", r21)

plt.scatter(exp, salary)
plt.plot(exp, model1.predict(exp))
plt.title("Experience vs Salary")
plt.show()


