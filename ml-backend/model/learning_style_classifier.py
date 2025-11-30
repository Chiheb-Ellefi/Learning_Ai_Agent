# ==========================================
# FILE 1: model/learning_style_classifier.py
# ==========================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class LearningStyleClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        """
        X: numpy array of shape (n_samples, 19 features)
        y: numpy array of learning style labels (0-3)
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train_scaled, y_train)
        test_acc = self.model.score(X_test_scaled, y_test)
        
        print(f"Training Accuracy: {train_acc:.3f}")
        print(f"Testing Accuracy: {test_acc:.3f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        return train_acc, test_acc, importances
    
    def predict(self, X):
        """Predict learning style for new user"""
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        return prediction[0], probabilities[0]
    
    def save_model(self, path="models/"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, f"{path}learning_style_model.pkl")
        joblib.dump(self.scaler, f"{path}learning_style_scaler.pkl")
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path="models/"):
        self.model = joblib.load(f"{path}learning_style_model.pkl")
        self.scaler = joblib.load(f"{path}learning_style_scaler.pkl")


def generate_synthetic_users(n_users=1000):
    """
    Generate synthetic user data for training
    Returns: X (features), y (labels)
    """
    print(f"Generating {n_users} synthetic users...")
    
    data = []
    
    for i in range(n_users):
        # Randomly assign a true learning style
        true_style = np.random.choice([0, 1, 2, 3])
        
        # Generate features based on learning style
        if true_style == 0:  # Visual
            features = {
                'avg_quiz_score': np.random.uniform(0.6, 0.9),
                'quiz_score_variance': np.random.uniform(0.05, 0.15),
                'avg_time_per_question': np.random.uniform(30, 60),
                'first_attempt_success_rate': np.random.uniform(0.6, 0.85),
                'improvement_rate': np.random.uniform(0.1, 0.3),
                
                'chat_to_reading_ratio': np.random.uniform(0.2, 0.5),
                'video_preference_ratio': np.random.uniform(0.7, 0.95),
                'article_preference_ratio': np.random.uniform(0.2, 0.5),
                'interactive_learning_score': np.random.uniform(0.6, 0.9),
                'self_study_score': np.random.uniform(0.5, 0.8),
                'discussion_engagement': np.random.uniform(0.3, 0.6),
                'practical_vs_theoretical': np.random.uniform(0.4, 0.7),
                
                'session_frequency': np.random.uniform(3, 7),
                'avg_session_duration': np.random.uniform(30, 90),
                'topic_completion_rate': np.random.uniform(0.6, 0.9),
                'revisit_frequency': np.random.uniform(0.2, 0.5),
                'struggle_indicator': np.random.uniform(0.2, 0.4),
                'learning_velocity': np.random.uniform(0.6, 0.9),
                'consistency_score': np.random.uniform(0.6, 0.9)
            }
        elif true_style == 1:  # Auditory
            features = {
                'avg_quiz_score': np.random.uniform(0.5, 0.85),
                'quiz_score_variance': np.random.uniform(0.08, 0.18),
                'avg_time_per_question': np.random.uniform(25, 55),
                'first_attempt_success_rate': np.random.uniform(0.5, 0.8),
                'improvement_rate': np.random.uniform(0.15, 0.35),
                
                'chat_to_reading_ratio': np.random.uniform(0.7, 0.95),
                'video_preference_ratio': np.random.uniform(0.5, 0.8),
                'article_preference_ratio': np.random.uniform(0.2, 0.5),
                'interactive_learning_score': np.random.uniform(0.7, 0.95),
                'self_study_score': np.random.uniform(0.3, 0.6),
                'discussion_engagement': np.random.uniform(0.7, 0.95),
                'practical_vs_theoretical': np.random.uniform(0.3, 0.6),
                
                'session_frequency': np.random.uniform(2, 6),
                'avg_session_duration': np.random.uniform(40, 100),
                'topic_completion_rate': np.random.uniform(0.5, 0.85),
                'revisit_frequency': np.random.uniform(0.3, 0.6),
                'struggle_indicator': np.random.uniform(0.3, 0.5),
                'learning_velocity': np.random.uniform(0.5, 0.8),
                'consistency_score': np.random.uniform(0.5, 0.85)
            }
        elif true_style == 2:  # Reading
            features = {
                'avg_quiz_score': np.random.uniform(0.7, 0.95),
                'quiz_score_variance': np.random.uniform(0.03, 0.12),
                'avg_time_per_question': np.random.uniform(40, 70),
                'first_attempt_success_rate': np.random.uniform(0.7, 0.9),
                'improvement_rate': np.random.uniform(0.08, 0.25),
                
                'chat_to_reading_ratio': np.random.uniform(0.1, 0.4),
                'video_preference_ratio': np.random.uniform(0.2, 0.5),
                'article_preference_ratio': np.random.uniform(0.7, 0.95),
                'interactive_learning_score': np.random.uniform(0.3, 0.6),
                'self_study_score': np.random.uniform(0.7, 0.95),
                'discussion_engagement': np.random.uniform(0.2, 0.5),
                'practical_vs_theoretical': np.random.uniform(0.6, 0.9),
                
                'session_frequency': np.random.uniform(4, 8),
                'avg_session_duration': np.random.uniform(50, 120),
                'topic_completion_rate': np.random.uniform(0.7, 0.95),
                'revisit_frequency': np.random.uniform(0.1, 0.4),
                'struggle_indicator': np.random.uniform(0.1, 0.3),
                'learning_velocity': np.random.uniform(0.7, 0.95),
                'consistency_score': np.random.uniform(0.7, 0.95)
            }
        else:  # Kinesthetic (3)
            features = {
                'avg_quiz_score': np.random.uniform(0.55, 0.85),
                'quiz_score_variance': np.random.uniform(0.1, 0.2),
                'avg_time_per_question': np.random.uniform(20, 50),
                'first_attempt_success_rate': np.random.uniform(0.5, 0.75),
                'improvement_rate': np.random.uniform(0.2, 0.4),
                
                'chat_to_reading_ratio': np.random.uniform(0.4, 0.7),
                'video_preference_ratio': np.random.uniform(0.5, 0.8),
                'article_preference_ratio': np.random.uniform(0.3, 0.6),
                'interactive_learning_score': np.random.uniform(0.8, 0.98),
                'self_study_score': np.random.uniform(0.4, 0.7),
                'discussion_engagement': np.random.uniform(0.5, 0.8),
                'practical_vs_theoretical': np.random.uniform(0.8, 0.98),
                
                'session_frequency': np.random.uniform(3, 7),
                'avg_session_duration': np.random.uniform(35, 95),
                'topic_completion_rate': np.random.uniform(0.55, 0.85),
                'revisit_frequency': np.random.uniform(0.4, 0.7),
                'struggle_indicator': np.random.uniform(0.25, 0.45),
                'learning_velocity': np.random.uniform(0.6, 0.85),
                'consistency_score': np.random.uniform(0.55, 0.85)
            }
        
        features['label'] = true_style
        data.append(features)
    
    df = pd.DataFrame(data)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"✅ Generated {n_users} users with {X.shape[1]} features")
    return X, y


