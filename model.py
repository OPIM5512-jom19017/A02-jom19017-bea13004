from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


# Load dataset
housing = fetch_california_housing(as_frame=True)


# Features and target
X = housing.data
y = housing.target


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


mlp = MLPRegressor(random_state=42,
                   hidden_layer_sizes=(10),
                   max_iter=200,
                   batch_size=1000,
                   activation="relu",
                   validation_fraction=0.2,
                   early_stopping=True) 

mlp.fit(X_train, y_train)

y_train_pred = mlp.predict(X_train)

plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_train_pred, alpha=0.3)
plt.plot([y_train.min(), y_train.max()],
[y_train.min(), y_train.max()],
linestyle="--")
plt.xlabel("Actual Values (Train)")
plt.ylabel("Predicted Values (Train)")
plt.title("MLP Regressor: Train Predictions vs Actual")
plt.show()