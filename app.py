# app.py
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.utils import secure_filename
from PIL import Image
from models import load_model, predict_food

# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my-secret-key-for-development'
# Database URI - supports both SQLite (default) and PostgreSQL (if DATABASE_URL is set)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(150))
    height_cm = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    conditions = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class FoodLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    food_name = db.Column(db.String(200))
    calories = db.Column(db.Float)
    protein = db.Column(db.Float, default=0)
    carbs = db.Column(db.Float, default=0)
    fats = db.Column(db.Float, default=0)
    serving_size = db.Column(db.String(100), default="100g")
    date = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(50))
    image_path = db.Column(db.String(300), nullable=True)

class ExerciseLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    exercise_name = db.Column(db.String(200))
    duration_minutes = db.Column(db.Float)  # Duration in minutes
    calories_burned = db.Column(db.Float)
    met_value = db.Column(db.Float)  # Metabolic Equivalent of Task
    date = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.String(500), nullable=True)

class ExerciseGoal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True)
    daily_calorie_goal = db.Column(db.Float, default=500)  # Daily calorie burn goal
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# Load model at startup
try:
    MODEL_PATH = os.path.join(BASE_DIR, 'food101_model_for_inference (1).pth')
    model, class_names, device = load_model(MODEL_PATH)
    print(f"Model loaded successfully with {len(class_names)} classes")
except Exception as e:
    model = None
    class_names = []
    device = None
    print(f"Model loading failed: {e}")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Extended Nutrition Database
NUTRITION_DB = {
    # Fruits
    'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fats': 0.2, 'serving': '100g'},
    'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fats': 0.3, 'serving': '100g'},
    'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fats': 0.1, 'serving': '100g'},
    'mango': {'calories': 60, 'protein': 0.8, 'carbs': 15, 'fats': 0.4, 'serving': '100g'},
    'grapes': {'calories': 69, 'protein': 0.7, 'carbs': 18, 'fats': 0.2, 'serving': '100g'},
    'watermelon': {'calories': 30, 'protein': 0.6, 'carbs': 8, 'fats': 0.2, 'serving': '100g'},
    'strawberry': {'calories': 32, 'protein': 0.7, 'carbs': 8, 'fats': 0.3, 'serving': '100g'},
    'pineapple': {'calories': 50, 'protein': 0.5, 'carbs': 13, 'fats': 0.1, 'serving': '100g'},
    'papaya': {'calories': 43, 'protein': 0.5, 'carbs': 11, 'fats': 0.3, 'serving': '100g'},
    'pomegranate': {'calories': 83, 'protein': 1.7, 'carbs': 19, 'fats': 1.2, 'serving': '100g'},
    
    # Vegetables
    'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fats': 0.4, 'serving': '100g'},
    'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fats': 0.2, 'serving': '100g'},
    'tomato': {'calories': 18, 'protein': 0.9, 'carbs': 4, 'fats': 0.2, 'serving': '100g'},
    'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 4, 'fats': 0.4, 'serving': '100g'},
    'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fats': 0.1, 'serving': '100g'},
    'onion': {'calories': 40, 'protein': 1.1, 'carbs': 9, 'fats': 0.1, 'serving': '100g'},
    'cucumber': {'calories': 15, 'protein': 0.7, 'carbs': 4, 'fats': 0.1, 'serving': '100g'},
    'cauliflower': {'calories': 25, 'protein': 1.9, 'carbs': 5, 'fats': 0.3, 'serving': '100g'},
    'bell pepper': {'calories': 31, 'protein': 1, 'carbs': 6, 'fats': 0.3, 'serving': '100g'},
    'cabbage': {'calories': 25, 'protein': 1.3, 'carbs': 6, 'fats': 0.1, 'serving': '100g'},
    
    # Grains & Staples
    'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fats': 0.3, 'serving': '100g cooked'},
    'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fats': 3.2, 'serving': '100g'},
    'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fats': 1.1, 'serving': '100g cooked'},
    'oats': {'calories': 389, 'protein': 17, 'carbs': 66, 'fats': 7, 'serving': '100g'},
    'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 21, 'fats': 1.9, 'serving': '100g cooked'},
    'wheat flour': {'calories': 364, 'protein': 10, 'carbs': 76, 'fats': 1, 'serving': '100g'},
    'corn': {'calories': 86, 'protein': 3.3, 'carbs': 19, 'fats': 1.4, 'serving': '100g'},
    
    # Proteins - Vegetarian
    'paneer': {'calories': 265, 'protein': 18, 'carbs': 1.2, 'fats': 20, 'serving': '100g'},
    'tofu': {'calories': 76, 'protein': 8, 'carbs': 1.9, 'fats': 4.8, 'serving': '100g'},
    'chickpeas': {'calories': 164, 'protein': 8.9, 'carbs': 27, 'fats': 2.6, 'serving': '100g cooked'},
    'lentils': {'calories': 116, 'protein': 9, 'carbs': 20, 'fats': 0.4, 'serving': '100g cooked'},
    'kidney beans': {'calories': 127, 'protein': 8.7, 'carbs': 23, 'fats': 0.5, 'serving': '100g cooked'},
    'black beans': {'calories': 132, 'protein': 8.9, 'carbs': 24, 'fats': 0.5, 'serving': '100g cooked'},
    'green peas': {'calories': 81, 'protein': 5, 'carbs': 14, 'fats': 0.4, 'serving': '100g'},
    'eggs': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fats': 11, 'serving': '100g'},
    'yogurt': {'calories': 59, 'protein': 10, 'carbs': 3.6, 'fats': 0.4, 'serving': '100g'},
    'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fats': 1, 'serving': '100ml'},
    
    # Proteins - Non-Vegetarian
    'chicken breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fats': 3.6, 'serving': '100g'},
    'chicken': {'calories': 239, 'protein': 27, 'carbs': 0, 'fats': 14, 'serving': '100g'},
    'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fats': 12, 'serving': '100g'},
    'salmon': {'calories': 208, 'protein': 20, 'carbs': 0, 'fats': 13, 'serving': '100g'},
    'tuna': {'calories': 130, 'protein': 28, 'carbs': 0, 'fats': 1, 'serving': '100g'},
    'shrimp': {'calories': 99, 'protein': 24, 'carbs': 0.2, 'fats': 0.3, 'serving': '100g'},
    'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fats': 15, 'serving': '100g'},
    'lamb': {'calories': 294, 'protein': 25, 'carbs': 0, 'fats': 21, 'serving': '100g'},
    'pork': {'calories': 242, 'protein': 27, 'carbs': 0, 'fats': 14, 'serving': '100g'},
    'turkey': {'calories': 189, 'protein': 29, 'carbs': 0, 'fats': 7, 'serving': '100g'},
    
    # Fast Food & Snacks
    'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fats': 10, 'serving': '100g'},
    'burger': {'calories': 295, 'protein': 17, 'carbs': 24, 'fats': 14, 'serving': '100g'},
    'french fries': {'calories': 312, 'protein': 3.4, 'carbs': 41, 'fats': 15, 'serving': '100g'},
    'sandwich': {'calories': 250, 'protein': 10, 'carbs': 30, 'fats': 10, 'serving': '100g'},
    'hot dog': {'calories': 290, 'protein': 10, 'carbs': 24, 'fats': 17, 'serving': '100g'},
    'tacos': {'calories': 217, 'protein': 9, 'carbs': 19, 'fats': 11, 'serving': '100g'},
    'nachos': {'calories': 312, 'protein': 7, 'carbs': 36, 'fats': 16, 'serving': '100g'},
    
    # Indian Foods
    'roti': {'calories': 297, 'protein': 11, 'carbs': 54, 'fats': 4, 'serving': '100g'},
    'naan': {'calories': 310, 'protein': 9, 'carbs': 52, 'fats': 7, 'serving': '100g'},
    'dosa': {'calories': 168, 'protein': 3.9, 'carbs': 28, 'fats': 4, 'serving': '100g'},
    'idli': {'calories': 156, 'protein': 4.4, 'carbs': 28, 'fats': 2.7, 'serving': '100g'},
    'biryani': {'calories': 200, 'protein': 8, 'carbs': 30, 'fats': 5, 'serving': '100g'},
    'dal': {'calories': 104, 'protein': 7, 'carbs': 17, 'fats': 1, 'serving': '100g'},
    'samosa': {'calories': 262, 'protein': 5, 'carbs': 32, 'fats': 13, 'serving': '100g'},
    'pakora': {'calories': 240, 'protein': 4, 'carbs': 22, 'fats': 15, 'serving': '100g'},
    'paratha': {'calories': 320, 'protein': 7, 'carbs': 42, 'fats': 14, 'serving': '100g'},
    
    # Nuts & Seeds
    'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fats': 50, 'serving': '100g'},
    'cashews': {'calories': 553, 'protein': 18, 'carbs': 30, 'fats': 44, 'serving': '100g'},
    'peanuts': {'calories': 567, 'protein': 26, 'carbs': 16, 'fats': 49, 'serving': '100g'},
    'walnuts': {'calories': 654, 'protein': 15, 'carbs': 14, 'fats': 65, 'serving': '100g'},
    'pumpkin seeds': {'calories': 446, 'protein': 19, 'carbs': 54, 'fats': 19, 'serving': '100g'},
    'sunflower seeds': {'calories': 584, 'protein': 21, 'carbs': 20, 'fats': 51, 'serving': '100g'},
    
    # Dairy & Alternatives
    'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fats': 33, 'serving': '100g'},
    'butter': {'calories': 717, 'protein': 0.9, 'carbs': 0.1, 'fats': 81, 'serving': '100g'},
    'cream': {'calories': 345, 'protein': 2.2, 'carbs': 2.7, 'fats': 37, 'serving': '100ml'},
    'ice cream': {'calories': 207, 'protein': 3.5, 'carbs': 24, 'fats': 11, 'serving': '100g'},
    
    # Beverages & Others
    'coffee': {'calories': 2, 'protein': 0.3, 'carbs': 0, 'fats': 0, 'serving': '100ml'},
    'tea': {'calories': 1, 'protein': 0, 'carbs': 0.3, 'fats': 0, 'serving': '100ml'},
    'orange juice': {'calories': 45, 'protein': 0.7, 'carbs': 10, 'fats': 0.2, 'serving': '100ml'},
    'soda': {'calories': 41, 'protein': 0, 'carbs': 11, 'fats': 0, 'serving': '100ml'},
    'honey': {'calories': 304, 'protein': 0.3, 'carbs': 82, 'fats': 0, 'serving': '100g'},
    'sugar': {'calories': 387, 'protein': 0, 'carbs': 100, 'fats': 0, 'serving': '100g'},
    'olive oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fats': 100, 'serving': '100ml'},
    'chocolate': {'calories': 546, 'protein': 5, 'carbs': 61, 'fats': 31, 'serving': '100g'},
    'cake': {'calories': 257, 'protein': 4.6, 'carbs': 41, 'fats': 9, 'serving': '100g'},
    'cookies': {'calories': 502, 'protein': 5.6, 'carbs': 64, 'fats': 25, 'serving': '100g'},
}

EXERCISE_DB = {
    # Cardio Exercises
    'walking_slow': {'name': 'Walking (Slow pace, 3 km/h)', 'met': 2.5, 'category': 'Cardio', 'icon': '🚶'},
    'walking_moderate': {'name': 'Walking (Moderate, 5 km/h)', 'met': 3.5, 'category': 'Cardio', 'icon': '🚶‍♂️'},
    'walking_brisk': {'name': 'Walking (Brisk, 6.5 km/h)', 'met': 5.0, 'category': 'Cardio', 'icon': '🚶‍♀️'},
    'jogging': {'name': 'Jogging (8 km/h)', 'met': 8.0, 'category': 'Cardio', 'icon': '🏃'},
    'running_moderate': {'name': 'Running (10 km/h)', 'met': 10.0, 'category': 'Cardio', 'icon': '🏃‍♂️'},
    'running_fast': {'name': 'Running (12 km/h)', 'met': 12.5, 'category': 'Cardio', 'icon': '🏃‍♀️'},
    'sprinting': {'name': 'Sprinting (16+ km/h)', 'met': 16.0, 'category': 'Cardio', 'icon': '💨'},
    
    # Cycling
    'cycling_leisure': {'name': 'Cycling (Leisure, 15 km/h)', 'met': 4.0, 'category': 'Cycling', 'icon': '🚴'},
    'cycling_moderate': {'name': 'Cycling (Moderate, 20 km/h)', 'met': 6.8, 'category': 'Cycling', 'icon': '🚴‍♂️'},
    'cycling_vigorous': {'name': 'Cycling (Vigorous, 25+ km/h)', 'met': 10.0, 'category': 'Cycling', 'icon': '🚴‍♀️'},
    'stationary_bike_light': {'name': 'Stationary Bike (Light)', 'met': 3.5, 'category': 'Cycling', 'icon': '🚴'},
    'stationary_bike_moderate': {'name': 'Stationary Bike (Moderate)', 'met': 6.8, 'category': 'Cycling', 'icon': '🚴‍♂️'},
    'stationary_bike_vigorous': {'name': 'Stationary Bike (Vigorous)', 'met': 10.0, 'category': 'Cycling', 'icon': '🚴‍♀️'},
    
    # Swimming
    'swimming_leisure': {'name': 'Swimming (Leisure)', 'met': 6.0, 'category': 'Swimming', 'icon': '🏊'},
    'swimming_laps_light': {'name': 'Swimming Laps (Light)', 'met': 7.0, 'category': 'Swimming', 'icon': '🏊‍♂️'},
    'swimming_laps_moderate': {'name': 'Swimming Laps (Moderate)', 'met': 8.0, 'category': 'Swimming', 'icon': '🏊‍♀️'},
    'swimming_laps_vigorous': {'name': 'Swimming Laps (Vigorous)', 'met': 10.0, 'category': 'Swimming', 'icon': '🏊'},
    
    # Strength Training
    'weight_training_light': {'name': 'Weight Training (Light)', 'met': 3.5, 'category': 'Strength', 'icon': '🏋️'},
    'weight_training_moderate': {'name': 'Weight Training (Moderate)', 'met': 5.0, 'category': 'Strength', 'icon': '🏋️‍♂️'},
    'weight_training_vigorous': {'name': 'Weight Training (Vigorous)', 'met': 6.0, 'category': 'Strength', 'icon': '🏋️‍♀️'},
    'bodyweight_exercises': {'name': 'Bodyweight Exercises', 'met': 5.0, 'category': 'Strength', 'icon': '💪'},
    'push_ups': {'name': 'Push-ups', 'met': 3.8, 'category': 'Strength', 'icon': '💪'},
    'pull_ups': {'name': 'Pull-ups', 'met': 8.0, 'category': 'Strength', 'icon': '💪'},
    
    # Sports
    'basketball': {'name': 'Basketball (Game)', 'met': 8.0, 'category': 'Sports', 'icon': '🏀'},
    'football': {'name': 'Football/Soccer', 'met': 7.0, 'category': 'Sports', 'icon': '⚽'},
    'tennis_singles': {'name': 'Tennis (Singles)', 'met': 8.0, 'category': 'Sports', 'icon': '🎾'},
    'tennis_doubles': {'name': 'Tennis (Doubles)', 'met': 6.0, 'category': 'Sports', 'icon': '🎾'},
    'badminton': {'name': 'Badminton', 'met': 5.5, 'category': 'Sports', 'icon': '🏸'},
    'volleyball': {'name': 'Volleyball', 'met': 4.0, 'category': 'Sports', 'icon': '🏐'},
    'cricket': {'name': 'Cricket', 'met': 4.8, 'category': 'Sports', 'icon': '🏏'},
    
    # HIIT & Aerobics
    'hiit': {'name': 'HIIT (High Intensity)', 'met': 12.0, 'category': 'HIIT', 'icon': '🔥'},
    'aerobics_low': {'name': 'Aerobics (Low Impact)', 'met': 5.0, 'category': 'Aerobics', 'icon': '🤸'},
    'aerobics_high': {'name': 'Aerobics (High Impact)', 'met': 7.0, 'category': 'Aerobics', 'icon': '🤸‍♂️'},
    'zumba': {'name': 'Zumba', 'met': 8.8, 'category': 'Aerobics', 'icon': '💃'},
    'dancing': {'name': 'Dancing (General)', 'met': 4.5, 'category': 'Aerobics', 'icon': '💃'},
    'jump_rope': {'name': 'Jump Rope', 'met': 12.3, 'category': 'HIIT', 'icon': '🦘'},
    'burpees': {'name': 'Burpees', 'met': 8.0, 'category': 'HIIT', 'icon': '🔥'},
    
    # Flexibility & Mind-Body
    'yoga_hatha': {'name': 'Yoga (Hatha)', 'met': 2.5, 'category': 'Flexibility', 'icon': '🧘'},
    'yoga_vinyasa': {'name': 'Yoga (Vinyasa)', 'met': 4.0, 'category': 'Flexibility', 'icon': '🧘‍♀️'},
    'yoga_power': {'name': 'Yoga (Power)', 'met': 4.0, 'category': 'Flexibility', 'icon': '🧘‍♂️'},
    'pilates': {'name': 'Pilates', 'met': 3.0, 'category': 'Flexibility', 'icon': '🧘'},
    'stretching': {'name': 'Stretching', 'met': 2.3, 'category': 'Flexibility', 'icon': '🤸'},
    
    # Other Activities
    'stairs_climbing': {'name': 'Stair Climbing', 'met': 8.8, 'category': 'Cardio', 'icon': '🪜'},
    'elliptical': {'name': 'Elliptical Machine', 'met': 5.0, 'category': 'Cardio', 'icon': '🏃'},
    'rowing': {'name': 'Rowing Machine', 'met': 7.0, 'category': 'Cardio', 'icon': '🚣'},
    'hiking': {'name': 'Hiking', 'met': 6.0, 'category': 'Outdoor', 'icon': '🥾'},
    'rock_climbing': {'name': 'Rock Climbing', 'met': 8.0, 'category': 'Outdoor', 'icon': '🧗'},
    'skating': {'name': 'Skating/Rollerblading', 'met': 7.0, 'category': 'Outdoor', 'icon': '⛸️'},
    'martial_arts': {'name': 'Martial Arts', 'met': 10.0, 'category': 'Sports', 'icon': '🥋'},
    'boxing': {'name': 'Boxing', 'met': 9.0, 'category': 'Sports', 'icon': '🥊'},
}


def calculate_calories_burned(exercise_key, duration_minutes, user_weight=70):
    """
    Calculate calories burned for an exercise
    Formula: Calories = MET × weight(kg) × time(hours)
    """
    if exercise_key not in EXERCISE_DB:
        return 0
    
    met = EXERCISE_DB[exercise_key]['met']
    hours = duration_minutes / 60.0
    calories = met * user_weight * hours
    
    return round(calories, 1)


def get_daily_exercise_stats(user_id):
    """Get today's exercise statistics"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_exercises = ExerciseLog.query.filter(
        ExerciseLog.user_id == user_id,
        ExerciseLog.date >= today_start
    ).all()
    
    total_calories = sum(ex.calories_burned or 0 for ex in today_exercises)
    total_duration = sum(ex.duration_minutes or 0 for ex in today_exercises)
    
    return {
        'total_calories': round(total_calories),
        'total_duration': round(total_duration),
        'exercise_count': len(today_exercises),
        'exercises': today_exercises
    }


# NEW ROUTES - Add these after the existing routes

@app.route('/exercise')
@login_required
def exercise():
    """Exercise tracking main page"""
    # Get or create exercise goal
    goal = ExerciseGoal.query.filter_by(user_id=current_user.id).first()
    if not goal:
        goal = ExerciseGoal(user_id=current_user.id, daily_calorie_goal=500)
        db.session.add(goal)
        db.session.commit()
    
    # Get today's stats
    stats = get_daily_exercise_stats(current_user.id)
    
    # Calculate progress
    progress_percentage = 0
    if goal.daily_calorie_goal > 0:
        progress_percentage = min(100, round((stats['total_calories'] / goal.daily_calorie_goal) * 100, 1))
    
    remaining_calories = max(0, goal.daily_calorie_goal - stats['total_calories'])
    
    # Group exercises by category
    exercises_by_category = {}
    for key, ex in EXERCISE_DB.items():
        category = ex['category']
        if category not in exercises_by_category:
            exercises_by_category[category] = []
        exercises_by_category[category].append({
            'key': key,
            'name': ex['name'],
            'met': ex['met'],
            'icon': ex['icon']
        })
    
    return render_template('exercise.html',
                         goal=goal,
                         stats=stats,
                         progress_percentage=progress_percentage,
                         remaining_calories=remaining_calories,
                         exercises_by_category=exercises_by_category)


@app.route('/log_exercise', methods=['POST'])
@login_required
def log_exercise():
    """Log a new exercise"""
    exercise_key = request.form.get('exercise_key')
    duration = float(request.form.get('duration', 0))
    notes = request.form.get('notes', '').strip()
    
    if not exercise_key or duration <= 0:
        flash("Please select an exercise and enter valid duration", "danger")
        return redirect(url_for('exercise'))
    
    if exercise_key not in EXERCISE_DB:
        flash("Invalid exercise selected", "danger")
        return redirect(url_for('exercise'))
    
    # Calculate calories burned
    user_weight = current_user.weight_kg or 70
    calories_burned = calculate_calories_burned(exercise_key, duration, user_weight)
    
    exercise_data = EXERCISE_DB[exercise_key]
    
    # Create log entry
    log = ExerciseLog(
        user_id=current_user.id,
        exercise_name=exercise_data['name'],
        duration_minutes=duration,
        calories_burned=calories_burned,
        met_value=exercise_data['met'],
        notes=notes
    )
    
    db.session.add(log)
    db.session.commit()
    
    flash(f"Logged {exercise_data['name']}: {duration} min, {calories_burned} kcal burned! 🔥", "success")
    return redirect(url_for('exercise'))


@app.route('/update_exercise_goal', methods=['POST'])
@login_required
def update_exercise_goal():
    """Update daily exercise goal"""
    new_goal = float(request.form.get('daily_goal', 500))
    
    goal = ExerciseGoal.query.filter_by(user_id=current_user.id).first()
    if not goal:
        goal = ExerciseGoal(user_id=current_user.id)
        db.session.add(goal)
    
    goal.daily_calorie_goal = new_goal
    goal.updated_at = datetime.utcnow()
    db.session.commit()
    
    flash(f"Exercise goal updated to {new_goal} kcal/day", "success")
    return redirect(url_for('exercise'))


@app.route('/exercise_history')
@login_required
def exercise_history():
    """View exercise history"""
    # Get all exercise logs
    logs = ExerciseLog.query.filter_by(user_id=current_user.id).order_by(ExerciseLog.date.desc()).all()
    
    # Calculate statistics
    total_calories = sum(log.calories_burned for log in logs)
    total_duration = sum(log.duration_minutes for log in logs)
    total_workouts = len(logs)
    
    # Get last 7 days stats
    week_ago = datetime.utcnow() - timedelta(days=7)
    week_logs = [log for log in logs if log.date >= week_ago]
    week_calories = sum(log.calories_burned for log in week_logs)
    week_duration = sum(log.duration_minutes for log in week_logs)
    
    # Daily average
    avg_daily_calories = round(week_calories / 7, 1) if week_logs else 0
    avg_daily_duration = round(week_duration / 7, 1) if week_logs else 0
    
    return render_template('exercise_history.html',
                         logs=logs,
                         total_calories=round(total_calories),
                         total_duration=round(total_duration),
                         total_workouts=total_workouts,
                         week_calories=round(week_calories),
                         week_duration=round(week_duration),
                         avg_daily_calories=avg_daily_calories,
                         avg_daily_duration=avg_daily_duration)


@app.route('/delete_exercise/<int:log_id>', methods=['POST'])
@login_required
def delete_exercise(log_id):
    """Delete an exercise log"""
    log = ExerciseLog.query.get_or_404(log_id)
    
    if log.user_id != current_user.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('exercise_history'))
    
    db.session.delete(log)
    db.session.commit()
    flash("Exercise log deleted", "success")
    return redirect(url_for('exercise_history'))

def lookup_nutrition(food_name):
    """Search for food in database with fuzzy matching"""
    key = food_name.lower().strip()
    
    # Exact match
    if key in NUTRITION_DB:
        return {'name': key, **NUTRITION_DB[key]}
    
    # Partial match
    for db_key in NUTRITION_DB:
        if db_key in key or key in db_key:
            return {'name': db_key, **NUTRITION_DB[db_key]}
    
    return None

def get_exercise_suggestions(calories, user_weight=70):
    """Calculate exercise suggestions based on calories and user weight"""
    # MET values (Metabolic Equivalent of Task)
    # Calories burned = MET × weight(kg) × time(hours)
    exercises = [
        {"name": "Walking (5 km/h)", "met": 3.5, "icon": "🚶"},
        {"name": "Jogging (8 km/h)", "met": 8.0, "icon": "🏃"},
        {"name": "Running (12 km/h)", "met": 12.0, "icon": "🏃‍♂️"},
        {"name": "Cycling (moderate)", "met": 6.8, "icon": "🚴"},
        {"name": "Swimming", "met": 8.0, "icon": "🏊"},
        {"name": "Jump Rope", "met": 12.3, "icon": "🦘"},
        {"name": "Yoga", "met": 3.0, "icon": "🧘"},
        {"name": "Weight Training", "met": 6.0, "icon": "🏋️"},
        {"name": "Dancing", "met": 4.5, "icon": "💃"},
        {"name": "Stairs Climbing", "met": 8.8, "icon": "🪜"},
    ]
    
    suggestions = []
    for ex in exercises:
        # Calories per minute = MET × weight(kg) × 0.0175
        cal_per_min = ex['met'] * user_weight * 0.0175
        minutes = max(1, int(round(calories / cal_per_min)))
        suggestions.append({
            'exercise': ex['name'],
            'minutes': minutes,
            'icon': ex['icon'],
            'calories_per_min': round(cal_per_min, 1)
        })
    
    return suggestions

def calculate_bmr(user):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation
    BMR (men) = 10 × weight(kg) + 6.25 × height(cm) − 5 × age(years) + 5
    BMR (women) = 10 × weight(kg) + 6.25 × height(cm) − 5 × age(years) − 161
    
    Returns daily calorie needs based on activity level (moderate activity multiplier = 1.55)
    """
    if not all([user.weight_kg, user.height_cm, user.age, user.gender]):
        return None
    
    # Calculate BMR
    bmr = 10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age
    
    if user.gender.lower() == 'male':
        bmr += 5
    elif user.gender.lower() == 'female':
        bmr -= 161
    else:
        # For 'other' or unspecified, use average
        bmr -= 78
    
    # Apply activity multiplier (moderate activity)
    # Sedentary: 1.2, Light: 1.375, Moderate: 1.55, Active: 1.725, Very Active: 1.9
    daily_calories = bmr * 1.55
    
    return round(daily_calories)


def get_daily_calorie_intake(user_id):
    """Get total calories consumed today"""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = FoodLog.query.filter(
        FoodLog.user_id == user_id,
        FoodLog.date >= today_start
    ).all()
    
    total_calories = sum(log.calories or 0 for log in today_logs)
    return round(total_calories), today_logs


# Update the /dashboard route (replace the existing one around line 195)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get logs for last 7 days
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_logs = FoodLog.query.filter(
        FoodLog.user_id == current_user.id,
        FoodLog.date >= week_ago
    ).order_by(FoodLog.date.desc()).all()
    
    # Calculate totals
    total_calories = sum(log.calories or 0 for log in recent_logs)
    total_protein = sum(log.protein or 0 for log in recent_logs)
    total_carbs = sum(log.carbs or 0 for log in recent_logs)
    total_fats = sum(log.fats or 0 for log in recent_logs)
    
    # Daily averages
    days_count = 7
    avg_calories = round(total_calories / days_count, 1)
    avg_protein = round(total_protein / days_count, 1)
    avg_carbs = round(total_carbs / days_count, 1)
    avg_fats = round(total_fats / days_count, 1)
    
    # BMI calculation
    bmi = None
    bmi_category = ""
    if current_user.height_cm and current_user.weight_kg:
        height_m = current_user.height_cm / 100.0
        if height_m > 0:
            bmi = round(current_user.weight_kg / (height_m * height_m), 1)
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
    
    # BMR and daily calorie tracking
    recommended_calories = calculate_bmr(current_user)
    consumed_today, today_logs = get_daily_calorie_intake(current_user.id)
    
    # Calculate percentage and remaining calories
    calorie_percentage = 0
    remaining_calories = 0
    if recommended_calories:
        calorie_percentage = round((consumed_today / recommended_calories) * 100, 1)
        remaining_calories = recommended_calories - consumed_today
    
    # NEW: Get today's exercise stats
    exercise_stats = get_daily_exercise_stats(current_user.id)
    
    # Get exercise goal
    exercise_goal = ExerciseGoal.query.filter_by(user_id=current_user.id).first()
    exercise_goal_value = exercise_goal.daily_calorie_goal if exercise_goal else 500
    
    # Calculate net calories (consumed - burned)
    net_calories = consumed_today - exercise_stats['total_calories']
    
    return render_template('dashboard.html',
                         recent_logs=recent_logs[:10],
                         total_calories=total_calories,
                         avg_calories=avg_calories,
                         avg_protein=avg_protein,
                         avg_carbs=avg_carbs,
                         avg_fats=avg_fats,
                         bmi=bmi,
                         bmi_category=bmi_category,
                         recommended_calories=recommended_calories,
                         consumed_today=consumed_today,
                         remaining_calories=remaining_calories,
                         calorie_percentage=calorie_percentage,
                         today_logs=today_logs,
                         # NEW: Exercise data
                         exercise_stats=exercise_stats,
                         exercise_goal=exercise_goal_value,
                         net_calories=net_calories)

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        name = request.form.get('name', '').strip()
        password = request.form['password']
        
        if User.query.filter_by(email=email).first():
            flash("Email already registered", "danger")
            return redirect(url_for('register'))
        
        user = User(email=email, name=name, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password", "danger")
            return redirect(url_for('login'))
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out", "info")
    return redirect(url_for('index'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        current_user.name = request.form.get('name', '').strip()
        current_user.height_cm = float(request.form.get('height_cm') or 0)
        current_user.weight_kg = float(request.form.get('weight_kg') or 0)
        current_user.age = int(request.form.get('age') or 0)
        current_user.gender = request.form.get('gender', '')
        current_user.conditions = request.form.get('conditions', '')
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for('profile'))
    
    return render_template('profile.html')

@app.route('/log_food', methods=['GET', 'POST'])
@login_required
def log_food():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = secure_filename(f"{current_user.id}_{datetime.now().timestamp()}_{file.filename}")
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                
                try:
                    img = Image.open(save_path).convert('RGB')
                    if model is None:
                        flash("AI model not available", "warning")
                        return redirect(url_for('log_food'))
                    
                    predictions = predict_food(model, img, class_names, device, topk=5)
                    return render_template('log_food.html',
                                         image_url=url_for('static', filename=f'uploads/{filename}'),
                                         predictions=predictions,
                                         saved_filename=filename)
                except Exception as e:
                    flash(f"Error processing image: {str(e)}", "danger")
                    return redirect(url_for('log_food'))
    
    return render_template('log_food.html')

@app.route('/accept_prediction', methods=['POST'])
@login_required
def accept_prediction():
    food_name = request.form['food_name']
    source = request.form.get('source', 'ai')
    image_path = request.form.get('image')
    
    nutrition = lookup_nutrition(food_name)
    if nutrition:
        log = FoodLog(
            user_id=current_user.id,
            food_name=nutrition['name'],
            calories=nutrition['calories'],
            protein=nutrition['protein'],
            carbs=nutrition['carbs'],
            fats=nutrition['fats'],
            serving_size=nutrition['serving'],
            source=source,
            image_path=image_path
        )
        db.session.add(log)
        db.session.commit()
        flash(f"Logged {nutrition['name']}: {nutrition['calories']} kcal", "success")
    else:
        flash("Food not found in database. Please use manual entry.", "warning")
    
    return redirect(url_for('dashboard'))

@app.route('/manual_entry', methods=['POST'])
@login_required
def manual_entry():
    food_name = request.form['food_name'].strip()
    calories = float(request.form.get('calories', 0))
    protein = float(request.form.get('protein', 0))
    carbs = float(request.form.get('carbs', 0))
    fats = float(request.form.get('fats', 0))
    serving_size = request.form.get('serving_size', '100g')
    
    log = FoodLog(
        user_id=current_user.id,
        food_name=food_name,
        calories=calories,
        protein=protein,
        carbs=carbs,
        fats=fats,
        serving_size=serving_size,
        source='manual'
    )
    db.session.add(log)
    db.session.commit()
    flash(f"Manually logged {food_name}", "success")
    return redirect(url_for('dashboard'))

@app.route('/search_food')
@login_required
def search_food():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'results': []})
    
    # Search in database
    results = []
    query_lower = query.lower()
    
    for food_key, nutrition in NUTRITION_DB.items():
        if query_lower in food_key or food_key in query_lower:
            results.append({
                'name': food_key,
                'calories': nutrition['calories'],
                'protein': nutrition['protein'],
                'carbs': nutrition['carbs'],
                'fats': nutrition['fats'],
                'serving': nutrition['serving']
            })
    
    return jsonify({'results': results[:10]})

@app.route('/food_history')
@login_required
def food_history():
    logs = FoodLog.query.filter_by(user_id=current_user.id).order_by(FoodLog.date.desc()).all()
    return render_template('food_history.html', logs=logs)

@app.route('/delete_log/<int:log_id>', methods=['POST'])
@login_required
def delete_log(log_id):
    log = FoodLog.query.get_or_404(log_id)
    if log.user_id != current_user.id:
        flash("Unauthorized", "danger")
        return redirect(url_for('food_history'))
    
    db.session.delete(log)
    db.session.commit()
    flash("Log deleted", "success")
    return redirect(url_for('food_history'))

@app.route('/exercise_suggestions', methods=['POST'])
@login_required
def exercise_suggestions():
    calories = float(request.form.get('calories', 0))
    user_weight = current_user.weight_kg or 70
    
    suggestions = get_exercise_suggestions(calories, user_weight)
    return jsonify({'calories': calories, 'suggestions': suggestions})


if __name__ == '__main__':
    app.run(debug=True)
