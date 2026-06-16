from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
# Load dataset
data = load_breast_cancer()
X, y = data.data , data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# # Display the dataset Information
# print("\n Features: ",data.feature_names)
# print("\n Classes: ",data.target_names)

# Train Random Forest 
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)

# Predict
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test,y_pred)
print(f" Random Forest Accuracy: {accuracy:.2f} \n Classification Report: {classification_report(y_test,y_pred)}")


# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50,100,200],
    'max_depth': [None,10,20],
    'max_features': ['sqrt','log2',None]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train,y_train)

# Display best parameters and score
print(f"Best Parameters: {grid_search.best_params_} \n Best Cross-Validation Accuracy: {grid_search.best_score_}")
