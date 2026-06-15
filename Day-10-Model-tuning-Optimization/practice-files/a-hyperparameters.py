from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

# Load datatse
data = load_breast_cancer()
X, y = data.data , data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# # Display dataset info
# print(f"Features : {data.feature_names}")
# print(f"Class Names: {data.target_names}")

# Train Random Forest with default hyperparameters
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train,y_train)

# Predict and evaluate
y_pred_default = rf_default.predict(X_test)
accuracy_default = accuracy_score(y_test,y_pred_default)
print(f"\nDefault Model Accuracy: {accuracy_default}")
print(f"Default Model Classification Report:\n{classification_report(y_test,y_pred_default)}")

# Train Random Forest with adjusted hyperparameters
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
rf_tuned.fit(X_train,y_train)

# Predict and evaluate
y_pred_tuned = rf_tuned.predict(X_test)
accuracy_tuned = accuracy_score(y_test,y_pred_tuned)
print(f"\nTuned Model Accuracy: {accuracy_tuned}")
print(f"Tuned Model Classification Report:\n{classification_report(y_test,y_pred_tuned)}")