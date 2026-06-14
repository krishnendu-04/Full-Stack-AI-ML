from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

# Load the dataset
data = load_breast_cancer()
X,y = data.data, data.target

# SPlit the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# # Display dataset information
# print("\nFeatures: ",data.feature_names)
# print("\nTarget Features: ",data.target_names)

# Train gradient boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train,y_train)

# Predict 
y_pred = gb_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test,y_pred)
print(f"Gradient Boosting Accuracy: {accuracy}")
print(f"classification report:\n {classification_report(y_test,y_pred)}")


# Define hyperparameter grid
param_grid = {
    'learning_rate': [0.01,0.1,0.2],
    'n_estimators' : [50,100,200],
    'max_depth' : [3,5,7]
}

# Perform grid search
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train,y_train)


# Display best paramters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")


# Train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)

# Predict
y_pred_rf = rf_model.predict(X_test)

# Evaluate
accuracy_rf = accuracy_score(y_test,y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")
print(f"classification report:\n {classification_report(y_test,y_pred_rf)}")
