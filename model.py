from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Load dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model pipeline with scaling and custom hyperparameters
model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        random_state=42,
        early_stopping=True,
        hidden_layer_sizes=(50, 25),     
        alpha=0.0005,                 
        learning_rate_init=0.001,         
        max_iter=500,
        batch_size=512,
        activation="relu",
        validation_fraction=0.2
    ))
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Train plot
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_train_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(f"Train Set: Actual vs Predicted")
plt.grid(True)
plt.savefig("figures/train_actual_vs_pred.png", dpi=300, bbox_inches="tight")
plt.show()

# Test plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title(f"Test Set: Actual vs Predicted")
plt.grid(True)
plt.savefig("figures/test_actual_vs_pred.png", dpi=300, bbox_inches="tight")
plt.show()
