from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the linear model
model = LinearRegression()
model.fit(X_train,y_train)

# PRedict
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

# Display the result
print(f"\n Mean Absolutw Error (MAE) : {mae:.2f} \n Mean Squared Error (MSE) : {mse:.2f} \n R2 - Score (R2) : {r2:.2f}")