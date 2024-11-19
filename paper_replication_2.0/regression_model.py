import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the data from pickle file
def load_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Analyze images using CLIP model from Hugging Face
def analyze_images_clip(image_folder_path, model, processor):
    image_features = []
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                # Extract image features from the vision encoder part of the CLIP model
                vision_outputs = model.get_image_features(**inputs)
                feature = vision_outputs.cpu().numpy().flatten()
                image_features.append(feature)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
    
    return np.array(image_features)

# Train a regression model
def train_model(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fitting a Random Forest Regressor model for better performance
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    # Predicting and evaluating the model
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    
    return regressor

# Main function to execute the script
def main():
    # Load the original data for richer feature set
    file_path = 'data_murty_185/all_data.pickle'  # Update the path if needed
    all_data = load_data(file_path)
    
    # Extract original features from the pickle data
    feature_keys = ['p1', 'p2', 'p3', 'p4']
    features = []
    for key in feature_keys:
        flattened_features = np.hstack([all_data[key][sub_key].reshape(len(all_data[key][sub_key]), -1) for sub_key in all_data[key].keys()])
        features.append(flattened_features)
    
    # Combining all feature sets into a single feature matrix
    original_features = np.hstack(features)
    
    # Choosing 'dp'['rffa'] as the target variable and flattening it to ensure consistency
    y = all_data['dp']['rffa']
    y = y.flatten()  # Flatten y to make it a 1-dimensional array
    
    # Align the number of samples in original features and target variable y
    num_samples = min(original_features.shape[0], len(y))
    X_aligned = original_features[:num_samples, :]
    y_aligned = y[:num_samples]
    
    # Load CLIP model from Hugging Face
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Analyze images using CLIP
    image_folder_path = 'data_murty_185/images_185'
    image_features = analyze_images_clip(image_folder_path, clip_model, processor)
    
    # Save the extracted image features for later use
    feature_save_path = 'image_features_resnet50.pkl'
    with open(feature_save_path, 'wb') as feature_file:
        pickle.dump(image_features, feature_file)
    
    print(f"Extracted and saved intermediate feature representations from images to {feature_save_path}")
    
    # Combine original features and image features
    if image_features.size > 0:
        # Align the number of samples in original features and image features
        num_samples = min(original_features.shape[0], image_features.shape[0])
        original_features_aligned = original_features[:num_samples, :]
        image_features_aligned = image_features[:num_samples, :]
        combined_features = np.hstack((original_features_aligned, image_features_aligned))
        y_aligned = y[:num_samples]
        
        # Train the model using combined features
        model = train_model(combined_features, y_aligned)
        
        # Save the trained model
        model_path = 'combined_regression_model.pkl'
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        
        # Predict the zones for the original data
        predicted_zones = model.predict(combined_features)
        print("Predicted Zones for Combined Data:")
        for idx, zone in enumerate(predicted_zones):
            print(f"Data Point {idx + 1}: Zone {zone}")
    else:
        print("No valid images found for analysis.")

if __name__ == "__main__":
    main()