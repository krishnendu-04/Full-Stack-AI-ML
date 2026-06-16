from sklearn.model_selection import train_test_split,RandomizedSearchCV,cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import numpy as np

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# # Display dataset info
# print("Dataset Info: \n",df.info())
# print(f"\nClass Distribution: {df['Churn'].value_counts()}")
# print("Sample Data: \n",df.head())

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna({'TotalCharges' :df['TotalCharges'].median()},inplace=True)

# Ecode the categorical variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Churn':
        df[col] = label_encoder.fit_transform(df[col])
        
        
# Encode the target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Scale the numerical features
scaler = StandardScaler()
numerical_features = ['tenure','MonthlyCharges','TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Features and Target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the initial model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train,y_train)

# Predict and evaluate initial model
y_pred = rf_model.predict(X_test)
accuracy_initial = accuracy_score(y_test,y_pred)

print(f"\nInitial Model Accuracy: {accuracy_initial:.4f}")
print(f"\n Classification Report (Initial Model): \n{classification_report(y_test,y_pred)}")

# Define the parameter grid
param_dist = {
    'n_estimators' : np.arange(50,200,10),
    'max_depth' : [None,5,10,15],
    'min_samples_split' : [2,5,10,20],
    'min_samples_leaf' : [1,2,4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)


# Perform Randomized Search
random_search.fit(X_train,y_train)

# Get the best parameters
best_params = random_search.best_params_
print(f"\nBest Parameters (RandomizedSearchCV): {best_params}")

# Train the best model
best_model = random_search.best_estimator_

# Predict and evaluate
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test,y_pred_tuned)

print(f"\nTuned Model Accuracy: {accuracy_tuned:.4f}")
print(f"\nClassification Report (Tuned Model):\n {classification_report(y_test,y_pred_tuned)}")

# Evaluate using cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")