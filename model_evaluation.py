import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the trained models
best_rf = joblib.load('../models/random_forest_model.pkl')
best_gb = joblib.load('../models/gradient_boosting_model.pkl')

# Load the preprocessed datasets
path = '../data/processed_data/'
X_test = joblib.load(path + 'X_test_preprocessed.pkl')
y_test = joblib.load(path + 'y_test_preprocessed.pkl')

# Predict using the Random Forest model
y_pred_rf = best_rf.predict(X_test)

# Predict using the Gradient Boosting model
y_pred_gb = best_gb.predict(X_test)

# Evaluate and print the performance of the Random Forest model
print("Random Forest:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_rf)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_rf)}")
print(f"R-squared: {r2_score(y_test, y_pred_rf)}")

# Evaluate and print the performance of the Gradient Boosting model
print("\nGradient Boosting:")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_gb)}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_gb)}")
print(f"R-squared: {r2_score(y_test, y_pred_gb)}")

# Plot the real vs. predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, y_pred_rf, color='blue', label='Random Forest', alpha=0.5)
plt.scatter(y_test, y_pred_gb, color='red', label='Gradient Boosting', alpha=0.5)
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Real vs. Predicted Stock Prices")
plt.legend()
plt.grid(True)
plt.show()
