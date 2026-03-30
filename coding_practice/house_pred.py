# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Step 1: Input data (House size in sq.ft)
# X = np.array([[500], [1000], [1500], [2000]])

# # Step 2: Output data (House price)
# y = np.array([100000, 200000, 300000, 400000])

# # Step 3: Create model
# model = LinearRegression()

# # Step 4: Train model
# model.fit(X, y)

# # Step 5: Predict
# predicted_price = model.predict([[1200]])

# print("Predicted Price:", predicted_price[0])


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# # Data
# X = np.array([[500], [1000], [1500], [2000]])
# y = np.array([100000, 200000, 300000, 400000])

# # Model
# model = LinearRegression()
# model.fit(X, y)

# # Predictions for line
# X_line = np.linspace(500, 2000, 100).reshape(-1, 1)
# y_line = model.predict(X_line)

# # Plot
# plt.scatter(X, y)        # actual points
# plt.plot(X_line, y_line) # regression line

# plt.xlabel("House Size")
# plt.ylabel("Price")
# plt.title("Linear Regression")

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data (not perfectly linear → realistic)
X = np.array([[500], [1000], [1500], [2000]])
y = np.array([1, 180000, 350000, 20])  # intentionally uneven

# Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Line for smooth plotting
X_line = np.linspace(500, 2000, 100).reshape(-1, 1)
y_line = model.predict(X_line)

# Plot actual points
plt.scatter(X, y)

# Plot regression line
plt.plot(X_line, y_line)

# Show error (residual lines)
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_pred[i]], linestyle='--')

plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("Wrong Predictions (Residuals)")

plt.show()

# from sklearn.linear_model import LogisticRegression

# # Step 1: Data (Hours → Pass/Fail)
# X = [[1], [2], [3], [4], [5]]   # input (hours studied)
# y = [0, 0, 0, 1, 1]            # output (0 = fail, 1 = pass)

# # Step 2: Create model
# model = LogisticRegression()

# # Step 3: Train model
# model.fit(X, y)

# # Step 4: Predict
# print("Study 3.5 hours:", model.predict([[-2]]))  # Expected: 0 (Fail)
# print("Study 0 hours:", model.predict([[0]]))  # Expected: 1 (Pass)