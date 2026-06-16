import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# Load dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# # Display dataset info
# print("Dataset Info: \n")
# print(df.info())
# print(f"\n Class Distribution: \n {df['Class'].value_counts()}")

# Define Features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initilaize K-Fold
kf = KFold(n_splits=5,shuffle=True,random_state=42)

# Train and evaluate the model
rf_model = RandomForestClassifier(random_state=42)
scores_kfold = cross_val_score(rf_model,X_train,y_train,cv=kf,scoring='accuracy')

print(f"K-Fold Cross Validation Scores: {scores_kfold}\n Mean Accuracy (K-Fold): {scores_kfold.mean():.2f}")

# Initialize the Stratified K-fold
skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

# Train and evaluate the model
scores_stratified = cross_val_score(rf_model,X_train,y_train,cv=skf,scoring='accuracy')

print(f"Stratified K-Fold Cross Validation Scores: {scores_stratified}\n Mean Accuracy (Stratified K-Fold): {scores_stratified.mean():.2f}")
