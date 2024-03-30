import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Data Preprocessing
df = pd.read_csv('./data-sets/yield_df.csv')

# Adjust label values
df['Item'] -= 1  # Subtract 1 from each label value to make them start from 0

# Perform one-hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Area'])  # Assuming 'Area' is the categorical variable

# Select features and target variable
X = df.drop(columns=['Item'])
y = df['Item']

# Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 2: Model Definition
model = Sequential([
    Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(units=64, activation='relu'),
    Dense(units=len(y.unique()), activation='softmax')
])

# Step 3: Model Compilation
opt = Adam(learning_rate=0.001)  # Set learning rate using learning_rate argument
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Step 4: Model Training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 5: Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")