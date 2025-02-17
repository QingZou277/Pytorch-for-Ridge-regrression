"""
Author: richardzou
Date: 0217,2025
"""
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# Scale data
scaler_X = StandardScaler()
X_train = torch.tensor(scaler_X.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32)

scaler_y = StandardScaler()
y_train = torch.tensor(scaler_y.fit_transform(y_train.reshape(-1,1)), dtype=torch.float32)
y_test = torch.tensor(scaler_y.transform(y_test.reshape(-1,1)), dtype=torch.float32)


# Define Ridge Regression model
class RidgeRegression(nn.Module):
    def __init__(self, input_dim, alpha):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

    def loss(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        l2_reg = self.alpha * torch.norm(self.linear.weight, p=2)
        return mse_loss + l2_reg

# Define Lasso Regression model
class LassoRegression(nn.Module):
    def __init__(self, input_dim, alpha):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.alpha = alpha

    def forward(self, x):
        return self.linear(x)

    def loss(self, output, target):
        mse_loss = nn.MSELoss()(output, target)
        l1_reg = self.alpha * torch.norm(self.linear.weight, p=1)
        return mse_loss + l1_reg

# Train the Ridge model
input_dim = X_train.shape[1]
alpha = 0.1  # You can adjust the regularization parameter
ridge_model = RidgeRegression(input_dim, alpha)
optimizer_ridge = optim.Adam(ridge_model.parameters(), lr=0.005)

num_epochs = 500

for epoch in range(num_epochs):
    optimizer_ridge.zero_grad()
    outputs = ridge_model(X_train)
    loss = ridge_model.loss(outputs, y_train)
    loss.backward()
    optimizer_ridge.step()
    if (epoch+1) % 10 == 0:
      print(f'Ridge Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Train the Lasso model
lasso_model = LassoRegression(input_dim, alpha)
optimizer_lasso = optim.SGD(lasso_model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    optimizer_lasso.zero_grad()
    outputs = lasso_model(X_train)
    loss = lasso_model.loss(outputs, y_train)
    loss.backward()
    optimizer_lasso.step()
    if (epoch+1) % 10 == 0:
      print(f'Lasso Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




# Predictions and evaluation for Ridge Regression
with torch.no_grad():
    ridge_predictions = ridge_model(X_test)
    ridge_mse = nn.MSELoss()(ridge_predictions, y_test)
    print(f'Ridge Regression MSE: {ridge_mse.item():.4f}')

# Predictions and evaluation for Lasso Regression
with torch.no_grad():
    lasso_predictions = lasso_model(X_test)
    lasso_mse = nn.MSELoss()(lasso_predictions, y_test)
    print(f'Lasso Regression MSE: {lasso_mse.item():.4f}')
