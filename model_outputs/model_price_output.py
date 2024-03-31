import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
# Load the saved model
model = load_model('./models/model_price.keras')

# Load training data to extract unique countries
df_train = pd.read_csv('./data-sets/8.csv')

# Define function to preprocess input data
def preprocess_input(input_data):
    # Define all possible products and entities based on training data
    all_products = ['Potatoes', 'Wheat', 'Rice', 'Sorghum', 'Soybeans', 'Cassava', 'Maize', 'Yams', 'Sweet Potatoes', 'Plantains']
    all_entities = df_train['Entity'].unique().tolist()  # Get unique countries from the training data
    
    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data)
    
    # Initialize one-hot encoded columns with zeros
    for product in all_products:
        input_df[f'Product_{product}'] = 0
    for entity in all_entities:
        input_df[f'Entity_{entity}'] = 0
    
    # Update one-hot encoded columns based on user input
    for key, value in input_data.items():
        if key.startswith('Product_'):
            product = key.split('_')[1]
            input_df[f'Product_{product}'] = value
        elif key.startswith('Entity_'):
            entity = key.split('_')[1]
            input_df[f'Entity_{entity}'] = value
    
    # Drop original columns ('Product' and 'Entity')
    input_df = input_df.drop(['Product', 'Entity'], axis=1, errors='ignore')
    
    # Normalize features using the same scaler used during training
    scaler = StandardScaler()
    input_normalized = scaler.fit_transform(input_df)
    
    return input_normalized

def predict_price(consumption, production, product_vegies, product_wheat,year,entity):
    user_input = {
    'Consumption': consumption,
    'Production': production,
    f'Product_{product_vegies}': [1],  # Example: One-hot encoding for the product Potatoes
    'Product_Wheat': product_wheat,     # Example: One-hot encoding for other products (not Wheat)
    # Add similar one-hot encoding columns for other products (if applicable)
    'Year': year,            # Example: Year 2024
    'Entity_Bulgaria': entity, # Example: One-hot encoding for the entity Afghanistan
    # Add similar one-hot encoding columns for other entities (if applicable)
    }
    # Preprocess user input
    input_normalized = preprocess_input(user_input)

    # Verify input shape
    expected_input_shape = model.input_shape[1:]  # Expected input shape of the model
    if input_normalized.shape[1] != expected_input_shape[0]:
        raise ValueError(f"Input shape mismatch: Expected {expected_input_shape}, but received {input_normalized.shape}")

    # Predict price for user input
    predictions = model.predict(input_normalized)

    return predictions[0][0]

# User input (example)


# Get predicted price

