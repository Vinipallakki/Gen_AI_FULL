# import numpy as np
# from sklearn.linear_model import LogisticRegression

# # Step 1: Input data (Hours studied)
# X = np.array([[1], [2], [3], [4], [5]])

# # Step 2: Output data (0 = Fail, 1 = Pass)
# y = np.array([0, 0, 0, 1, 1])

# # Step 3: Create model
# model = LogisticRegression()

# # Step 4: Train model
# model.fit(X, y)

# # Step 5: Predict
# prediction = model.predict([[3.5]])

# print("Prediction (0=Fail, 1=Pass):", prediction[0])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data (Hours vs Pass/Fail)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])  # 0 = Fail, 1 = Pass

# Model
model = LogisticRegression()
model.fit(X, y)

# Smooth curve
X_curve = np.linspace(0, 6, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_curve)[:, 1]

# Decision boundary (0.5 probability)
decision_boundary = X_curve[np.where(y_prob >= 0.5)[0][0]]

# Plot actual points
plt.scatter(X, y)

# Plot sigmoid curve
plt.plot(X_curve, y_prob)

# Plot decision boundary
plt.axvline(x=decision_boundary, linestyle='--')

plt.xlabel("Hours Studied")
plt.ylabel("Probability / Class")
plt.title("Pass vs Fail (Logistic Regression)")

plt.show()