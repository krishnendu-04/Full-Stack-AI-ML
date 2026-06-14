import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select features and target
features = ['Pclass','Sex','Age','Fare','Embarked']
target = 'Survived'

# Handle missing values
df.fillna({'Age':df['Age'].median()},inplace=True)
df.fillna({'Embarked':df['Embarked'].mode()},inplace=True)

# ENcode categorical variables
label_encoders = {}
for col in ['Sex','Embarked']:
    le =LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# Split the data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Training Data Shape: {X_train.shape} ")
# print(f"Test Data Shape: {X_test.shape}")

# Train the lightGBM model
lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train,y_train)

# Predict and evaluate
lgb_pred = lgb_model.predict(X_test)
print(f"\nLightGBM Accuracy: {accuracy_score(y_test,lgb_pred)}")

# Train CatBoost Model
cat_features = ['Pclass','Sex','Embarked',]
cat_model = CatBoostClassifier(cat_features=cat_features,verbose=0)
cat_model.fit(X_train,y_train)

# Predict and evaluate
cat_pred = cat_model.predict(X_test)
print(f"\nCatBoost Accuracy: {accuracy_score(y_test,cat_pred)}")


# Train xgboost model
xgb_model = XGBClassifier(eval_metric='logloss')
xgb_model.fit(X_train,y_train)

# Predict and evaluate
xgb_pred = xgb_model.predict(X_test)
print(f"\nXGBoost Accuracy: {accuracy_score(y_test,xgb_pred)}")


# Train catboost without encoding categorical features
cat_model_native = CatBoostClassifier(cat_features=['Sex','Embarked'],verbose=0)
cat_model_native.fit(X_train,y_train)

# Predict and evaluate
cat_pred_native = cat_model_native.predict(X_test)
print(f"CatBoost Native Accuracy: {accuracy_score(y_test,cat_pred_native)}")