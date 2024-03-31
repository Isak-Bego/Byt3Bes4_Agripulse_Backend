import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Step 1: Data Preprocessing
df = pd.read_csv('../data-sets/yield_df.csv')

# Select features and target variable
X = df[['Area', 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']]  # Features
y = df['Item']  # Target variable

# Perform one-hot encoding for categorical columns
X = pd.get_dummies(X, columns=['Area'])

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 2: Model Definition
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(units=1)  # Linear output layer for regression
])

# Step 3: Model Compilation
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)

# Step 4: Model Training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 5: Model Evaluation
loss = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")

# Step 6: Save the trained model in native Keras format
save_model(model, '../models/model_product.keras')
print("Regularized model saved successfully.")