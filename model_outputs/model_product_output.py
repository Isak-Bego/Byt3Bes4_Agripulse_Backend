import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('../models/model_product.keras')

# Load training data to extract unique countries
df_train = pd.read_csv('../data-sets/yield_df.csv')

# Define function to preprocess input data
def preprocess_input(input_data):
    # Define all possible products and entities based on training data
    all_entities = df_train['Area'].unique().tolist()  # Get unique countries from the training data
    
    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data)
    
    # Initialize one-hot encoded columns with zeros
    for entity in all_entities:
        input_df[f'Area_{entity}'] = 0
    
    # Update one-hot encoded columns based on user input
    for key, value in input_data.items():
        if key.startswith('Area_'):
            entity = key.split('_')[1]
            input_df[f'Area_{entity}'] = value
    
    # Drop original columns ('Area')
    input_df = input_df.drop(['Area'], axis=1, errors='ignore')
    
    # Normalize features using the same scaler used during training
    scaler = StandardScaler()
    input_normalized = scaler.fit_transform(input_df)
    
    return input_normalized

def switch_case(item_number):
    switcher = {
        1: "Maize",
        2: "Potatoes",
        3: "Rice, paddy",
        4: "Sorghum",
        5: "Soybeans",
        6: "Wheat",
        7: "Cassava",
        8: "Sweet potatoes",
        9: "Plantains and others",
        10: "Yams"
    }

    return switcher.get(item_number, "Invalid case")

def find_adjacent_items(user_input):
    # Preprocess user input
    input_normalized = preprocess_input(user_input)

    # Predict item for user input
    predicted_item = model.predict(input_normalized)
    item_number = round(predicted_item[0][0])

    # Calculate adjacent item numbers
    lower_item_number = max(1, item_number - 1)
    higher_item_number = min(10, item_number + 1)

    # Get item names for adjacent item numbers
    lower_item = switch_case(lower_item_number)
    higher_item = switch_case(higher_item_number)

    # Get current item
    current_item = switch_case(item_number)

    return [lower_item, current_item, higher_item]

# Example usage
user_input = {
    'Area_Bulgaria': [1],  # Example value
    'hg/ha_yield': [36213],
    'average_rain_fall_mm_per_year': [1455],  
    'pesticides_tonnes': [11],     
    'avg_temp': [20.2],            
}

adjacent_items = find_adjacent_items(user_input)
print("Adjacent Items:", adjacent_items)
