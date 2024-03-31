import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load a sample of the dataset
df = pd.read_csv('./data-sets/8.csv').sample(n=1000, random_state=42)  # Adjust the sample size as needed

# Select features and target variable
X = df[['Price', 'Population', 'Production', 'Product', 'Year', 'Entity']]  # Features
y = df['Consumption']  # Target variable

# Perform one-hot encoding for categorical columns
X = pd.get_dummies(X, columns=['Product', 'Entity'])

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {mae}")