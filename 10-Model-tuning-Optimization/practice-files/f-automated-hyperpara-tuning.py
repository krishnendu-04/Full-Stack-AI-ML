from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.svm import SVC
import numpy as np

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define param grid
param_grid = {
    'n_estimators' : [50,100,150],
    'learning_rate' : [0.01,0.1,0.2],
    'max_depth' : [3,5,7],
}

# Initilaize the GridSearchCV
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

# Perform Grid Search
grid_search.fit(X_train,y_train)

# Get best parameters and score
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_
print(f"Best Parameters for GridSearchCV: {best_params_grid}\nBest Cross-Validation Accuracy for GridSearchCV: {best_score_grid}")

# Get best model
best_grid_model = grid_search.best_estimator_

# Predict and evaluate
y_pred_grid = best_grid_model.predict(X_test)
accuracy_grid = accuracy_score(y_test,y_pred_grid)

print(f"Test Accuracy (GridSearchCV): {accuracy_grid}")
print(f"\nClassification Report: \n{classification_report(y_test,y_pred_grid)}")

# Define parameter distribution
param_dist = {
    'C' : np.logspace(-3,3,10),
    'kernel' : ['linear','rbf','poly','sigmoid'],
    'gamma' : ['scale','auto']
}

# Initilaize the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# Perform Randomized Search
random_search.fit(X_train,y_train)

# Get best parameters and score
best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"Best Parameters for RandomizedSearchCV: {best_params_random}\nBest Cross-Validation Accuracy for RandomizedSearchCV: {best_score_random}")


# Get best model
best_random_model = random_search.best_estimator_

# Predict and evaluate
y_pred_random = best_random_model.predict(X_test)
accuracy_random = accuracy_score(y_test,y_pred_random)

print(f"Test Accuracy (RandomizedSearchCV): {accuracy_random}")
print(f"\nClassification Report: \n{classification_report(y_test,y_pred_random)}")