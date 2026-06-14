import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report,roc_auc_score

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# # Display dataset info and preview
# print(f"Dataset Info: \n {df.info()}\n Class Distribution: \n {df['Churn'].value_counts()} \n Sample Data: \n {df.head()}  ")

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna({'TotalCharges':df['TotalCharges'].median()},inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Churn':
        df[column] = label_encoder.fit_transform(df[column])
        
        
# Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Scale numerical features
scaler = StandardScaler()
numerical_features = ['tenure','MonthlyCharges','TotalCharges']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Define features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Handle Imbalance Data
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)
# # Display class distribution after SMOTE
# print(f"\n Class Distribution After SMOTE: \n{pd.Series(y_train_resampled).value_counts()}")

# Train Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled,y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
roc_auc_rf = roc_auc_score(y_test,rf_model.predict_proba(X_test)[:,1])

# Train XGBOOST
xgb_model = XGBClassifier(eval_metric='logloss',random_state=42)
xgb_model.fit(X_train_resampled,y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
roc_auc_xgb = roc_auc_score(y_test,xgb_model.predict_proba(X_test)[:,1])

# Train LightGBM
lgbm_model = LGBMClassifier(random_state=42)
lgbm_model.fit(X_train_resampled,y_train_resampled)
y_pred_lgbm = lgbm_model.predict(X_test)
roc_auc_lgbm = roc_auc_score(y_test,lgbm_model.predict_proba(X_test)[:,1])


# Classification reports
print(f"Random Forest Report: \n{classification_report(y_test,y_pred_rf)}")
print(f"XGBoost Report: \n{classification_report(y_test,y_pred_xgb)}")
print(f"LightGBM Report: \n{classification_report(y_test,y_pred_lgbm)}")

# ROC_AUC comparison
print("ROC-AUC Scores:\n")
print(f"Random Forest : {roc_auc_rf}")
print(f"XGBoost : {roc_auc_xgb}")
print(f"LightGBM : {roc_auc_lgbm}")