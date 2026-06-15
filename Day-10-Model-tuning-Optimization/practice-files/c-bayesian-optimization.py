from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import optuna

# Load dataset
data = load_breast_cancer()
X, y = data.data , data.target

# # Display dataset info
# print(f"Features: {data.feature_names}\n Target/Classes : {data.target_names}")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
# print(f"Training data shape: {X_train.shape}\nTesting data shape: {X_test.shape}")

# Train xgboost model
baseline_model = XGBClassifier(eval_metric='logloss',random_state=42)
baseline_model.fit(X_train,y_train)

# Predict and evaluate the model
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test,baseline_pred)
print(f"Baesline XGBoost Accuracy: {baseline_accuracy}")

# Define objective function for optuna
def objective(trial):
    params ={
        'n_estimators' : trial.suggest_int('n_estimators',50,500),
        'max_depth' : trial.suggest_int('max_depth',3,100),
        'learning_rate' : trial.suggest_float('learning_rate',0.01,50.3),
        'subsample' : trial.suggest_float('subsample',0.6,1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree',0.6,1.0),
        'gamma': trial.suggest_float('gamma',0,5),
        'reg_alpha': trial.suggest_float('reg_alpha',0,10),
        'reg_lambda': trial.suggest_float('reg_lambda',0,10),
    }
    
    # Train xgboost model with suggested params
    model = XGBClassifier(eval_metric='logloss',random_state=42,**params)
    model.fit(X_train,y_train)
    
    # Evaluate model on validation set
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,pred)
    return accuracy


# Create an Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=50)

# Best hyperparameters
print(f"Best Hyperparameters: {study.best_params}\n Best Accuracy: {study.best_value}")


# Define parameter grid
param_grid = {
    'n_estimators' : [100,200,300],
    'max_depth' : [3,5,7],
    'learning_rate' : [0.01,0.1,0.2],
    'subsample' : [0.6,0.8,1.0]
}


# Train the XGBoost with gridsearch
grid_search = GridSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    verbose=1
)
grid_search.fit(X_train,y_train)

# Best parameters and accuracy
print(f"Grid Search Best Parameters: {grid_search.best_params_}\n Grid Search Best Accuracy: {grid_search.best_score_}")

# Define parameter distribution
param_dist = {
    'n_estimators' : [50,100,200,300,400],
    'max_depth' : [3,5,7,9],
    'learning_rate' : [0.01,0.05,0.1,0.2],
    'subsample' : [0.6,0.7,0.8,0.9,1.0],
    'colsample_bytree' : [0.6,0.7,0.8,0.9,1.0]
}

# Train the XGBoost with Random Search 
random_search = RandomizedSearchCV(
    estimator=XGBClassifier(eval_metric='logloss',random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)
random_search.fit(X_train,y_train)


# Best parameters and accuracy
print(f"Random Search Best Parameters: {random_search.best_params_}\n Random Search Best Accuracy: {random_search.best_score_}")
