
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target

# Validate dataset
assert X.shape[0] > 0, "Dataset is empty"
assert len(set(y)) == 3, "Expected three classes"

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Validate results
assert 0.0 <= accuracy <= 1.0
assert len(predictions) == len(y_test)

# Save model
joblib.dump(model, "model.pkl")

# Generate build report
with open("build_report.txt", "w", encoding="utf-8") as report:
    report.write("Machine Learning Build Report\n")
    report.write("==============================\n")
    report.write("Dataset: Iris\n")
    report.write("Algorithm: Random Forest\n")
    report.write(f"Training samples: {len(X_train)}\n")
    report.write(f"Testing samples: {len(X_test)}\n")
    report.write(f"Accuracy: {accuracy:.4f}\n")
    report.write("Training status: SUCCESSFUL\n")

print("Model training completed successfully.")
print(f"Test accuracy: {accuracy:.4f}")
print("Generated model.pkl and build_report.txt")
