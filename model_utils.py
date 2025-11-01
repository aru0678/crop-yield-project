import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CropYieldPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the crop yield dataset"""
        df = pd.read_csv(csv_path)
        
        # Drop Date column (not needed for regression)
        df = df.drop(columns=["Date"])
        
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=["Crop_Type", "Soil_Type"], drop_first=True)
        
        # Define features and target
        X = df_encoded.drop("Crop_Yield", axis=1)
        y = df_encoded["Crop_Yield"]
        
        self.feature_columns = X.columns.tolist()
        
        return X, y, df
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.is_trained = True
        
        return {
            'r2_score': r2,
            'rmse': rmse,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict(self, features):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if isinstance(features, dict):
            # Convert dict to DataFrame with proper feature ordering
            features_df = pd.DataFrame([features])
            # Ensure all required columns are present
            for col in self.feature_columns:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_columns]
            return self.model.predict(features_df)[0]
        else:
            return self.model.predict(features)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True

def get_yield_insights(features, predicted_yield):
    """Get LLM-generated insights for crop yield predictions"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        return "API key not configured for insights generation"
    
    prompt = f"""
    The predicted crop yield is {predicted_yield:.2f} units per hectare.
    Conditions:
    - Soil pH: {features.get('Soil_pH', 'N/A')}
    - Temperature: {features.get('Temperature', 'N/A')} °C
    - Humidity: {features.get('Humidity', 'N/A')} %
    - Wind Speed: {features.get('Wind_Speed', 'N/A')} km/h
    - Nitrogen (N): {features.get('N', 'N/A')}
    - Phosphorus (P): {features.get('P', 'N/A')}
    - Potassium (K): {features.get('K', 'N/A')}
    - Soil Quality: {features.get('Soil_Quality', 'N/A')}
    
    Based on these conditions, provide:
    1. Brief analysis of the predicted yield
    2. Key factors affecting this yield
    3. 3-5 actionable recommendations to improve yield
    
    Keep the response concise and practical for farmers.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                               headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error generating insights: {response.status_code}"
    except Exception as e:
        return f"Error connecting to insights service: {str(e)}"

def generate_sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    
    crop_types = ['Wheat', 'Corn', 'Rice', 'Barley', 'Soybean']
    soil_types = ['Loamy', 'Sandy', 'Clay', 'Peaty']
    
    sample_data = []
    for i in range(100):
        data = {
            'Soil_pH': np.random.uniform(5.0, 8.0),
            'Temperature': np.random.uniform(10, 35),
            'Humidity': np.random.uniform(40, 90),
            'Wind_Speed': np.random.uniform(2, 15),
            'N': np.random.uniform(20, 100),
            'P': np.random.uniform(15, 80),
            'K': np.random.uniform(10, 70),
            'Soil_Quality': np.random.uniform(20, 80),
            'Crop_Type': np.random.choice(crop_types),
            'Soil_Type': np.random.choice(soil_types)
        }
        sample_data.append(data)
    
    return pd.DataFrame(sample_data)