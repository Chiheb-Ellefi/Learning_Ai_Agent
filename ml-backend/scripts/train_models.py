# scripts/train_models.py
import sys
import os

# Add the parent directory (which is the project root containing the 'model' folder)
# to the Python search path. This allows 'from model...' to work.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.learning_style_classifier import LearningStyleClassifier, generate_synthetic_users
from model.quiz_performance_predictor import QuizPerformancePredictor, generate_quiz_training_data

# Train Learning Style Classifier
print("Training Learning Style Classifier...")
style_clf = LearningStyleClassifier()
X_style, y_style = generate_synthetic_users(1000)
train_acc, test_acc, importance = style_clf.train(X_style, y_style)
style_clf.save_model()
print(f"✅ Model saved! Test Accuracy: {test_acc:.3f}")

# Train Quiz Performance Predictor
print("\nTraining Quiz Performance Predictor...")
quiz_pred = QuizPerformancePredictor()
X_quiz, y_quiz = generate_quiz_training_data(500)
importance = quiz_pred.train(X_quiz, y_quiz)
quiz_pred.save_model()
print("✅ Model saved!")

print("\n🎉 All models trained and ready to use!")
