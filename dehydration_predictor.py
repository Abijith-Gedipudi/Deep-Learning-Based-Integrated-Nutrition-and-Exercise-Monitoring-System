# dehydration_predictor.py
# ML-based Dehydration Risk Prediction System

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class DehydrationPredictor:
    """
    Machine Learning model to predict dehydration risk
    Uses daily habits, weather, and activity data
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'water_intake_ml',
            'urination_frequency',
            'activity_level_encoded',
            'temperature_celsius',
            'humidity_percent',
            'outdoor_duration_minutes',
            'hour_of_day',
            'body_weight_kg',
            'age',
            'gender_encoded',
            'caffeine_intake_ml',
            'alcohol_intake_ml',
            'exercise_calories_burned'
        ]
        
    def generate_synthetic_training_data(self, n_samples=5000):
        """
        Generate synthetic training data based on medical research
        This simulates real-world patterns for model training
        """
        np.random.seed(42)
        
        data = []
        
        for _ in range(n_samples):
            # Base features
            weight = np.random.normal(70, 15)  # kg
            age = np.random.randint(18, 80)
            gender = np.random.choice([0, 1])  # 0=female, 1=male
            
            # Calculate recommended water intake
            base_water = weight * 35  # 35ml per kg
            if gender == 0:  # Female
                base_water *= 0.95
            
            # Activity level affects water needs
            activity = np.random.choice([0, 1, 2])  # low, medium, high
            activity_multiplier = [1.0, 1.3, 1.6][activity]
            recommended_water = base_water * activity_multiplier
            
            # Actual water intake (varies around recommended)
            intake_ratio = np.random.normal(0.85, 0.25)  # Most people under-hydrate
            water_intake = recommended_water * intake_ratio
            water_intake = max(500, min(5000, water_intake))  # Realistic bounds
            
            # Urination frequency (4-8 times normal, affected by intake)
            base_urination = 6
            urination = base_urination + (water_intake - 2000) / 500
            urination += np.random.normal(0, 1)
            urination = max(2, min(15, urination))
            
            # Weather conditions
            temperature = np.random.uniform(15, 40)  # Celsius
            humidity = np.random.uniform(30, 90)  # Percent
            
            # Outdoor exposure
            outdoor_duration = np.random.exponential(60)  # Minutes
            outdoor_duration = min(480, outdoor_duration)  # Max 8 hours
            
            # Time of day
            hour = np.random.randint(0, 24)
            
            # Other factors
            caffeine = np.random.exponential(100) if np.random.random() > 0.3 else 0
            caffeine = min(500, caffeine)
            
            alcohol = np.random.exponential(100) if np.random.random() > 0.7 else 0
            alcohol = min(500, alcohol)
            
            exercise_calories = [0, 200, 500][activity] + np.random.normal(0, 100)
            exercise_calories = max(0, exercise_calories)
            
            # Calculate dehydration risk based on multiple factors
            risk_score = 0
            
            # Water intake deficit
            deficit = recommended_water - water_intake
            if deficit > 500:
                risk_score += 2
            elif deficit > 200:
                risk_score += 1
            
            # Low urination frequency
            if urination < 4:
                risk_score += 2
            elif urination < 5:
                risk_score += 1
            
            # High temperature
            if temperature > 30:
                risk_score += 2
            elif temperature > 25:
                risk_score += 1
            
            # Low humidity
            if humidity < 40:
                risk_score += 1
            
            # Extended outdoor exposure
            if outdoor_duration > 180:
                risk_score += 2
            elif outdoor_duration > 90:
                risk_score += 1
            
            # Caffeine/alcohol (diuretics)
            if caffeine > 300:
                risk_score += 1
            if alcohol > 200:
                risk_score += 2
            
            # High activity without sufficient water
            if activity == 2 and water_intake < recommended_water * 0.9:
                risk_score += 2
            
            # Determine risk category
            if risk_score >= 7:
                risk = 2  # High
            elif risk_score >= 4:
                risk = 1  # Moderate
            else:
                risk = 0  # Low
            
            data.append({
                'water_intake_ml': water_intake,
                'urination_frequency': urination,
                'activity_level_encoded': activity,
                'temperature_celsius': temperature,
                'humidity_percent': humidity,
                'outdoor_duration_minutes': outdoor_duration,
                'hour_of_day': hour,
                'body_weight_kg': weight,
                'age': age,
                'gender_encoded': gender,
                'caffeine_intake_ml': caffeine,
                'alcohol_intake_ml': alcohol,
                'exercise_calories_burned': exercise_calories,
                'risk_level': risk
            })
        
        return pd.DataFrame(data)
    
    def train_model(self, df=None):
        """Train the dehydration prediction model"""
        
        if df is None:
            print("Generating synthetic training data...")
            df = self.generate_synthetic_training_data(n_samples=5000)
        
        X = df[self.feature_names]
        y = df['risk_level']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return self.model
    
    def predict(self, water_intake_ml, urination_events, activity_level, 
                temperature_c, humidity_percent, outdoor_exposure_minutes, 
                hour_of_day, body_weight_kg=70, age=30, gender=1, 
                caffeine_intake_ml=0, alcohol_intake_ml=0, exercise_calories_burned=0):
        """
        Predict dehydration risk - simplified interface for app integration
        
        Parameters:
        -----------
        water_intake_ml : float
        urination_events : int
        activity_level : int (0=low, 1=medium, 2=high)
        temperature_c : float
        humidity_percent : float
        outdoor_exposure_minutes : float
        hour_of_day : int (0-23)
        body_weight_kg : float (default 70)
        age : int (default 30)
        gender : int (0=female, 1=male, default 1)
        caffeine_intake_ml : float (default 0)
        alcohol_intake_ml : float (default 0)
        exercise_calories_burned : float (default 0)
        
        Returns:
        --------
        dict : prediction results with risk level, confidence, and recommendations
        """
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare features dictionary
        features = {
            'water_intake_ml': float(water_intake_ml),
            'urination_frequency': float(urination_events),
            'activity_level_encoded': int(activity_level),
            'temperature_celsius': float(temperature_c),
            'humidity_percent': float(humidity_percent),
            'outdoor_duration_minutes': float(outdoor_exposure_minutes),
            'hour_of_day': int(hour_of_day),
            'body_weight_kg': float(body_weight_kg),
            'age': int(age),
            'gender_encoded': int(gender),
            'caffeine_intake_ml': float(caffeine_intake_ml),
            'alcohol_intake_ml': float(alcohol_intake_ml),
            'exercise_calories_burned': float(exercise_calories_burned)
        }
        
        # Prepare feature vector in correct order
        feature_vector = [features[name] for name in self.feature_names]
        
        # Scale and predict
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        risk_level = self.model.predict(X_scaled)[0]
        risk_probabilities = self.model.predict_proba(X_scaled)[0]
        
        risk_labels = {0: 'Low', 1: 'Moderate', 2: 'High'}
        
        # Generate recommendations
        recommendations = self._generate_recommendations(features, risk_level)
        
        # Generate risk/protective factors analysis
        analysis = self._analyze_factors(features, risk_level)
        
        return {
            'risk_level': risk_labels[risk_level],
            'risk_score': int(risk_level),
            'probabilities': {
                'Low': round(float(risk_probabilities[0]), 3),
                'Moderate': round(float(risk_probabilities[1]), 3),
                'High': round(float(risk_probabilities[2]), 3)
            },
            'confidence': round(float(max(risk_probabilities)), 3),
            'recommendations': recommendations,
            'analysis': analysis
        }
    
    def _analyze_factors(self, features, risk_level):
        """Analyze risk and protective factors"""
        
        risk_factors = []
        protective_factors = []
        
        # Calculate recommended water intake
        recommended_water = features['body_weight_kg'] * 35
        if features['gender_encoded'] == 0:
            recommended_water *= 0.95
        
        activity_multiplier = [1.0, 1.3, 1.6][features['activity_level_encoded']]
        recommended_water *= activity_multiplier
        
        # Check water intake
        deficit = recommended_water - features['water_intake_ml']
        if deficit > 500:
            risk_factors.append(f"Significant water deficit ({int(deficit)}ml below recommendation)")
        elif deficit < -500:
            protective_factors.append(f"Excellent water intake ({int(abs(deficit))}ml above recommendation)")
        elif deficit < 0:
            protective_factors.append("Adequate water intake")
        
        # Urination frequency
        if features['urination_frequency'] < 4:
            risk_factors.append(f"Low urination frequency ({features['urination_frequency']:.0f} times/day)")
        elif features['urination_frequency'] >= 6:
            protective_factors.append(f"Healthy urination frequency ({features['urination_frequency']:.0f} times/day)")
        
        # Temperature
        if features['temperature_celsius'] > 30:
            risk_factors.append(f"High ambient temperature ({features['temperature_celsius']:.1f}°C)")
        elif features['temperature_celsius'] < 20:
            protective_factors.append(f"Moderate temperature ({features['temperature_celsius']:.1f}°C)")
        
        # Humidity
        if features['humidity_percent'] < 40:
            risk_factors.append(f"Low humidity ({features['humidity_percent']:.0f}%)")
        elif features['humidity_percent'] > 60:
            protective_factors.append(f"Good humidity level ({features['humidity_percent']:.0f}%)")
        
        # Activity level
        if features['activity_level_encoded'] >= 2:
            risk_factors.append("High physical activity level")
        elif features['activity_level_encoded'] == 0:
            protective_factors.append("Low activity reduces dehydration risk")
        
        # Outdoor exposure
        if features['outdoor_duration_minutes'] > 180:
            risk_factors.append(f"Extended outdoor exposure ({features['outdoor_duration_minutes']:.0f} minutes)")
        elif features['outdoor_duration_minutes'] < 60:
            protective_factors.append("Limited outdoor exposure")
        
        # Caffeine and alcohol
        if features['caffeine_intake_ml'] > 300:
            risk_factors.append(f"High caffeine intake ({features['caffeine_intake_ml']:.0f}ml)")
        
        if features['alcohol_intake_ml'] > 200:
            risk_factors.append(f"High alcohol intake ({features['alcohol_intake_ml']:.0f}ml)")
        elif features['alcohol_intake_ml'] == 0:
            protective_factors.append("No alcohol consumption")
        
        return {
            'risk_factors': risk_factors,
            'protective_factors': protective_factors
        }
    
    def _generate_recommendations(self, features, risk_level):
        """Generate personalized hydration recommendations"""
        
        recommendations = []
        
        # Calculate water deficit
        recommended_water = features['body_weight_kg'] * 35
        if features['gender_encoded'] == 0:
            recommended_water *= 0.95
        
        activity_multiplier = [1.0, 1.3, 1.6][features['activity_level_encoded']]
        recommended_water *= activity_multiplier
        
        deficit = recommended_water - features['water_intake_ml']
        
        # Priority recommendations based on risk level
        if risk_level == 2:  # High risk
            recommendations.append({
                'priority': 'URGENT',
                'message': 'High dehydration risk detected',
                'action': 'Drink 500ml of water immediately. Rest and avoid strenuous activity.',
                'icon': '🚨'
            })
        
        # Water intake recommendations
        if deficit > 500:
            hourly_amount = int(deficit / 4)
            recommendations.append({
                'priority': 'HIGH',
                'message': f'Water deficit: {int(deficit)}ml below daily goal',
                'action': f'Drink {hourly_amount}ml every hour for the next 4 hours',
                'icon': '💧'
            })
        elif deficit > 200:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'Slightly low water intake ({int(deficit)}ml deficit)',
                'action': 'Drink a full glass of water now (250ml)',
                'icon': '💦'
            })
        elif deficit < -500:
            recommendations.append({
                'priority': 'LOW',
                'message': 'Excellent hydration status!',
                'action': 'Maintain current water intake throughout the day',
                'icon': '✅'
            })
        
        # Urination frequency
        if features['urination_frequency'] < 4:
            recommendations.append({
                'priority': 'HIGH',
                'message': 'Low urination frequency indicates dehydration',
                'action': 'Increase water intake significantly. Monitor urine color (should be pale yellow)',
                'icon': '⚠️'
            })
        
        # Weather-based
        if features['temperature_celsius'] > 28:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'High temperature: {features["temperature_celsius"]:.0f}°C',
                'action': 'Add 500ml extra water. Stay in shade when possible',
                'icon': '☀️'
            })
        
        if features['humidity_percent'] < 40:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'Low humidity: {features["humidity_percent"]:.0f}%',
                'action': 'Increase water intake by 250-500ml due to faster evaporation',
                'icon': '🌵'
            })
        
        # Activity-based
        if features['activity_level_encoded'] >= 2:
            recommendations.append({
                'priority': 'HIGH',
                'message': 'High activity level detected',
                'action': 'Drink 200-300ml water every 20 minutes during exercise',
                'icon': '🏃'
            })
        
        # Outdoor exposure
        if features['outdoor_duration_minutes'] > 120:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'Extended outdoor time: {features["outdoor_duration_minutes"]:.0f} minutes',
                'action': 'Carry water bottle. Drink 250ml every 30 minutes outdoors',
                'icon': '🌞'
            })
        
        # Caffeine/Alcohol
        if features['caffeine_intake_ml'] > 300:
            recommendations.append({
                'priority': 'MEDIUM',
                'message': f'High caffeine intake: {features["caffeine_intake_ml"]:.0f}ml (diuretic effect)',
                'action': 'Add 500ml extra water to compensate for fluid loss',
                'icon': '☕'
            })
        
        if features['alcohol_intake_ml'] > 200:
            recommendations.append({
                'priority': 'HIGH',
                'message': f'Alcohol consumption: {features["alcohol_intake_ml"]:.0f}ml',
                'action': 'Drink 250ml water for each alcoholic beverage consumed',
                'icon': '🍺'
            })
        
        # General maintenance
        if risk_level == 0 and not recommendations:
            recommendations.append({
                'priority': 'LOW',
                'message': 'Good hydration status maintained',
                'action': 'Continue drinking water regularly throughout the day',
                'icon': '✅'
            })
        
        return recommendations[:6]  # Limit to top 6 recommendations
    
    def save_model(self, filepath='dehydration_model.pkl'):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath='dehydration_model.pkl'):
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        print(f"✓ Model loaded from {filepath}")


class WeatherAPI:
    """
    Fetch real-time weather data for dehydration prediction
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "6414e928de2aa52a15d1ceb20ef94695"
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.default_weather = {
            'temperature_c': 25.0,
            'humidity_percent': 60.0,
            'weather': 'Clear',
            'feels_like': 25.0,
            'city': 'Unknown'
        }
    
    def get_weather(self, city="Hyderabad", lat=None, lon=None):
        """
        Get current weather data
        
        Returns dict with: temperature_c, humidity_percent, weather, feels_like, city
        """
        
        params = {
            'appid': self.api_key,
            'units': 'metric'
        }
        
        if lat and lon:
            params['lat'] = lat
            params['lon'] = lon
        else:
            params['q'] = city
        
        try:
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            return {
                'temperature_c': round(float(data['main']['temp']), 1),
                'humidity_percent': float(data['main']['humidity']),
                'weather': data['weather'][0]['description'],
                'feels_like': round(float(data['main']['feels_like']), 1),
                'city': data['name']
            }
        
        except Exception as e:
            print(f"Weather API error: {e}. Using default values.")
            return self.default_weather.copy()


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("ML-Based Dehydration Risk Prediction System")
    print("=" * 60)
    
    # Initialize and train model
    predictor = DehydrationPredictor()
    predictor.train_model()
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("Testing Prediction")
    print("=" * 60)
    
    # Test prediction
    result = predictor.predict(
        water_intake_ml=1200,
        urination_events=3,
        activity_level=2,
        temperature_c=32,
        humidity_percent=45,
        outdoor_exposure_minutes=180,
        hour_of_day=14
    )
    
    print(f"\nRisk Level: {result['risk_level']}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"\nRecommendations ({len(result['recommendations'])}):")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. [{rec['priority']}] {rec['icon']} {rec['message']}")