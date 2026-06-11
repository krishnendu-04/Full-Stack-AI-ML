# Task 1:Perform EDA and Preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
data = fetch_california_housing(as_frame=True)
df=data.frame

X = df[['MedInc','HouseAge','AveRooms']]
y = df['MedHouseVal']

# # Inspect Data
# print(df.info())
# print(df.describe())

# # Visualize relationships
# sns.pairplot(df,vars=['MedInc','AveRooms','HouseAge','MedHouseVal'])
# plt.show()
# #Check for missing values
# print("Missing Values: \n",df.isnull().sum())

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the linear regression model
model = LinearRegression()
model.fit(X_train,y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate performance
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error: {mse}")