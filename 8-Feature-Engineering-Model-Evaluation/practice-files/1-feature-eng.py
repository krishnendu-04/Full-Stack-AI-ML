import pandas as pd

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# #display dataset information
# print("\nDataset Info:\n")
# print(df.info())

# # Preview the first few rows
# print("\nDataset Preview:\n")
# print(df.head())

# Seperate Features
categorical_features = df.select_dtypes(include=["object"]).columns
numerical_features = df.select_dtypes(include=["int64","float64"]).columns

print("\nCategorical Features: \n ",categorical_features.to_list())
print("\nNumerical Features: \n",numerical_features.to_list())


# Display summary of categorical features
print("\n Categorical Feature Summary: \n")
for col in categorical_features:
    print(f"{col}:\n", df[col].value_counts(),"\n")
    
# Display summary of numerical features
print("\n Numerical Feature Summary: \n")
print(df[numerical_features].describe())