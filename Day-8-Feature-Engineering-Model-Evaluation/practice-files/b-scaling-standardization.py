from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

# Load Iris datset
data = load_iris()
X = pd.DataFrame(data.data,columns = data.feature_names)
y = data.target


# Display dataset information
print("\n Dataset Info:\n")
print(X.describe())
print("\n Target Classes: ",data.target_names)

# Split Dataset
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#Predict and evaluate
y_pred = knn.predict(X_test)
print("\n Accuracy Without Scaling: ",accuracy_score(y_test,y_pred))


# Apply Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#Split scaled data
X_trained_scaled, X_test_scaled, y_train_scaled,y_test_scaled = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# Train k-NN CLassifier on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_trained_scaled,y_train_scaled)

#Predict and evaluate
y_pred_scaled = knn_scaled.predict(X_test_scaled)
print("\n Accuracy with Min-Max Scaling: ",accuracy_score(y_test_scaled,y_pred_scaled))


# Apply Standardization
scaler2 = StandardScaler()
X_std = scaler2.fit_transform(X)

#Split standardized data
X_trained_std, X_test_std, y_train_std,y_test_std = train_test_split(X_std,y,test_size=0.2,random_state=42)

# Train k-NN CLassifier on standardized data
knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_trained_std,y_train_std)

#Predict and evaluate
y_pred_std = knn_std.predict(X_test_std)
print("\n Accuracy with Standardization: ",accuracy_score(y_test_std,y_pred_std))