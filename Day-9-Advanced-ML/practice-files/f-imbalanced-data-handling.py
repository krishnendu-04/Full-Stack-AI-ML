import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,roc_auc_score
from imblearn.over_sampling import SMOTE

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Explore the dataset
print("\n Dataset Info:\n ",df.info(),"\nClass Distribution: \n",df["Class"].value_counts())

# Split the dataset
X = df.drop(columns=['Class'])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

# Train the model
rf_model = RandomForestClassifier(random_state=42,class_weight="balanced")
rf_model.fit(X_train,y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
print(f"\nClassification Report: \n{classification_report(y_test,y_pred)}")

roc_auc = roc_auc_score(y_test,rf_model.predict_proba(X_test)[:,1])
print(f"ROC-AUC: {roc_auc}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train,y_train)

# Display the new class distribution
print("\n Class Distribution after SMOTE: \n")
print(pd.Series(y_resampled).value_counts())


# Train Random Forest on resampled data
rf_model_smote = RandomForestClassifier(random_state=42)
rf_model_smote.fit(X_resampled,y_resampled)

# Predict and evaluate
y_pred_smote = rf_model_smote.predict(X_test)
print(f"\nClassification Report(SMOTE) : \n{classification_report(y_test,y_pred_smote)}")

roc_auc_smote = roc_auc_score(y_test,rf_model_smote.predict_proba(X_test)[:,1])
print(f"ROC-AUC (SMOTE) : {roc_auc_smote}")
