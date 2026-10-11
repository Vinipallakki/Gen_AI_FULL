# # Inputs
# study_hours = 8
# attendance = 80

# # Weights
# w1 = 0.5
# w2 = 0.1

# # Bias
# bias = -10

# # Calculate weighted sum
# z = (study_hours * w1) + (attendance * w2) + bias  #z=w1​x1​+w2​x2​+b

# print("Weighted Sum:", z)

# # Step activation function
# if z >= 0:
#     prediction = 1
# else:
#     prediction = 0

# print("Prediction:", prediction)

# if prediction == 1:
#     print("Student will PASS")
# else:
#     print("Student will FAIL")

import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------------------------------
# 1. Training data
# --------------------------------------------------

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
])


# --------------------------------------------------
# 2. Create the model
# --------------------------------------------------

model = nn.Linear(2, 1)


# --------------------------------------------------
# 3. Loss function
# --------------------------------------------------

loss_function = nn.MSELoss()


# --------------------------------------------------
# 4. Optimizer
# --------------------------------------------------

optimizer = optim.SGD(
    model.parameters(),
    lr=0.1
)


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

for epoch in range(1000):

    # Forward pass
    prediction = model(X)

    # Calculate loss
    loss = loss_function(prediction, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights and bias
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, Loss: {loss.item():.4f}"
        )


# --------------------------------------------------
# 6. Check learned weights and bias
# --------------------------------------------------

print("\nLearned Weight:")
print(model.weight)

print("\nLearned Bias:")
print(model.bias)


# --------------------------------------------------
# 7. Test the model
# --------------------------------------------------

with torch.no_grad():

    predictions = model(X)

    print("\nRaw Predictions:")
    print(predictions)