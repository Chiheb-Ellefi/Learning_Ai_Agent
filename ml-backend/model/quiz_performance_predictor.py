# ==========================================
# FILE 2: model/quiz_performance_predictor.py
# ==========================================

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class QuizPerformancePredictor:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
    
    def prepare_features(self, user_data):
        """Extract features from user data"""
        prev_scores = user_data.get('previous_scores', [])
        
        features = [
            np.mean(prev_scores) if prev_scores else 0.5,
            np.std(prev_scores) if len(prev_scores) > 1 else 0,
            prev_scores[-1] if prev_scores else 0.5,
            len(prev_scores),
            user_data.get('topic_difficulty', 0.5),
            user_data.get('days_since_last_quiz', 1),
            user_data.get('total_study_time', 60),
            user_data.get('chat_messages_sent', 0),
            user_data.get('resources_viewed', 1),
            self.encode_learning_style(user_data.get('learning_style', 'Visual')),
            user_data.get('total_study_time', 60) / max(user_data.get('days_since_last_quiz', 1), 1),
            user_data.get('chat_messages_sent', 0) / max(user_data.get('resources_viewed', 1), 1),
        ]
        
        return np.array(features).reshape(1, -1)
    
    def encode_learning_style(self, style):
        styles = {'Visual': 0, 'Auditory': 1, 'Reading': 2, 'Kinesthetic': 3}
        return styles.get(style, 0)
    
    def train(self, X, y):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        print("✅ Quiz Performance Predictor trained successfully")
        return self.model.feature_importances_
    
    def predict(self, user_data):
        """Predict quiz score"""
        features = self.prepare_features(user_data)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        confidence = 0.1
        
        return {
            'predicted_score': float(prediction),
            'lower_bound': float(max(0, prediction - confidence)),
            'upper_bound': float(min(1, prediction + confidence)),
            'advice': self.get_advice(prediction)
        }
    
    def get_advice(self, predicted_score):
        if predicted_score >= 0.8:
            return "You're well prepared! You should do great on this quiz."
        elif predicted_score >= 0.6:
            return "You're on the right track. Review key concepts before starting."
        else:
            return "Consider spending more time studying this topic before taking the quiz."
    
    def save_model(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f"{path}quiz_predictor.pkl")
        joblib.dump(self.scaler, f"{path}quiz_scaler.pkl")
        print(f"✅ Quiz predictor saved to {path}")
    
    def load_model(self, path="models/"):
        self.model = joblib.load(f"{path}quiz_predictor.pkl")
        self.scaler = joblib.load(f"{path}quiz_scaler.pkl")


def generate_quiz_training_data(n_samples=500):
    """Generate synthetic quiz data"""
    print(f"Generating {n_samples} quiz samples...")
    
    data = []
    
    for i in range(n_samples):
        num_prev_quizzes = np.random.randint(0, 10)
        base_ability = np.random.uniform(0.4, 0.9)
        
        prev_scores = [
            np.clip(base_ability + np.random.normal(0, 0.1), 0, 1)
            for _ in range(num_prev_quizzes)
        ]
        
        topic_difficulty = np.random.uniform(0.3, 0.9)
        days_since = np.random.randint(0, 14)
        study_time = np.random.randint(30, 300)
        chat_msgs = np.random.randint(0, 20)
        resources = np.random.randint(2, 15)
        learning_style = np.random.randint(0, 4)
        
        avg_score = np.mean(prev_scores) if prev_scores else 0.5
        std_score = np.std(prev_scores) if len(prev_scores) > 1 else 0
        last_score = prev_scores[-1] if prev_scores else 0.5
        experience = len(prev_scores)
        study_intensity = study_time / max(days_since, 1)
        engagement = chat_msgs / max(resources, 1)
        
        features = [
            avg_score, std_score, last_score, experience,
            topic_difficulty, days_since, study_time,
            chat_msgs, resources, learning_style,
            study_intensity, engagement
        ]
        
        true_score = (
            base_ability * 0.5 +
            avg_score * 0.3 +
            (1 - topic_difficulty) * 0.1 +
            min(study_time / 180, 0.2) +
            -min(days_since / 14, 0.1) +
            np.random.normal(0, 0.05)
        )
        true_score = np.clip(true_score, 0, 1)
        
        data.append({
            'features': features,
            'score': true_score
        })
    
    X = np.array([d['features'] for d in data])
    y = np.array([d['score'] for d in data])
    
    print(f"✅ Generated {n_samples} quiz samples with {X.shape[1]} features")
    return X, y


